import time
import sys
import os
import numpy as np
import json
import glob
import argparse
import torch.utils.checkpoint
import torch.nn.functional as F
from matplotlib import pyplot as plt
import sys

sys.path.append('/work2/08264/baagee/frontera/gns-main/')
from meshnet import utils
from meshnet import data_loader
from meshnet import learned_simulator
from meshnet import visualizer
from example.inverse_meshnet.forward import rollout_with_checkpointing
from example.inverse_meshnet import tools


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="/work2/08264/baagee/frontera/gns-main/example/inverse_meshnet/config.json", type=str,
                    help="Path to input json file (e.g., `data/config.json`")
args = parser.parse_args()


# Read the JSON configuration file
with open(args.input_path, 'r') as file:
    config = json.load(file)

# Set paths
data_path = "/work2/08264/baagee/frontera/gns-main/example/inverse_meshnet/data/"
output_path = "/work2/08264/baagee/frontera/gns-main/example/inverse_meshnet/outputs/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
save_step = 1

# Inputs for ground truth
ground_truth_npz = "porous0.npz"
lx, ly = 200, 200
is_fixed_mesh = True

# Inputs for optimizer
optimizer_type = "adam"  # or lbfgs
niteration = 50
inverse_timestep_range = [300, 350]
observation_patches_edge = [
    [5, 90],
    [10, 50], [10, 140],
    [20, 20], [20, 110], [20, 120], [20, 160],
    [30, 40], [30, 140],
    [40, 180],
    [50, 80],
    [60, 25], [60, 60], [50, 135], [60, 170],
    [70, 115],
    [90, 55], [90, 90], [90, 140],
    [100, 150],
    [115, 15], [115, 85],
    [120, 150]
]
observation_patches_size = [10, 10]
checkpoint_interval = 1
lr = 0.01
initial_vleft_x = tools.vel_autogen(
    ly=ly, shape_option="uniform", args={"peak": [0.25, 0.25], "npoints": ly})

# inputs for forward simulator
simulator_metadata = utils.read_metadata(data_path, "rollout", "metadata-7gnn.json")
model_file = "model-6150000.pt"
n_message_passing = simulator_metadata['nmessage_passing_steps'] if simulator_metadata is not None else 10
node_type_embedding_size = 9
dt = 0.01  # unnecessary since it is not used in model
INPUT_SEQUENCE_LENGTH = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# inputs for output setup
save_interval = 1

# resume
resume = False
resume_iteration = 0

# Inputs for visualizing
lx_physical = 1.0
ly_physical = lx_physical * ly/lx
x_conversion = lx_physical/lx
y_conversion = ly_physical/ly


# load simulator
simulator = learned_simulator.MeshSimulator(
    simulation_dimensions=2,
    nnode_in=11,
    nedge_in=3,
    latent_dim=128,
    nmessage_passing_steps=n_message_passing,
    nmlp_layers=2,
    mlp_hidden_dim=128,
    nnode_types=3,
    node_type_embedding_size=9,
    device=device)
simulator.load(f"{data_path}/{model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth velocity at the inversion timesteps and node coordinates
ds = data_loader.get_data_loader_by_trajectories(
    path=f"{data_path}/{ground_truth_npz}", fixed_mesh=is_fixed_mesh)
features = next(iter(ds))
node_coords = features[0].to(device)  # (timesteps, nnode, ndims)
node_type = features[1].to(device)  # (timesteps, nnode, )
velocities = features[2].to(device)  # (timesteps, nnode, ndims)
cells = features[3].to(device)

# transform velocity to velocity grid, and get the target velocity
ntimesteps = len(velocities)
# Note that node_coords follows column-based order. Therefore, we need to the following conversion
vel_fields_grid_true = velocities.clone().reshape(ntimesteps, ly, lx, 2).permute((0, 2, 1, 3))
# Empty the initial vel-x that will be updated during optimization
vel_fields_grid_true[0, 0, :, 0] = 0
# Get the target data at the inverse timestep and spacial domain range
target_vel_fields_patches = tools.sample_patches(
    vel_fields_grid_true,
    inverse_timestep_range[0], inverse_timestep_range[1],
    observation_patches_size[0], observation_patches_size[1],
    patch_locations=observation_patches_edge
)

# Initialize initial x-velocity
initial_vleft_x = torch.tensor(initial_vleft_x, requires_grad=True, device=device)
# Resister as torch model parameter so that it can be passed to optimizer object
initial_vleft_x_model = tools.To_Torch_Model_Param(initial_vleft_x)

# Set up the optimizer
if optimizer_type == "lbfgs":
    optimizer = torch.optim.LBFGS(initial_vleft_x_model.parameters(), lr=lr, max_iter=4)
elif optimizer_type == "adam":
    optimizer = torch.optim.Adam(initial_vleft_x_model.parameters(), lr=lr)
elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(initial_vleft_x_model.parameters(), lr=lr)
else:
    raise ValueError("Check `optimizer_type`")

# Save the current config to the output dir
with open(f"{output_path}/config.json", "w") as outfile:
    json.dump(config, outfile, indent=4)

# Resume
if resume:
    print(f"Resume from the previous state: iteration{resume_iteration}")
    checkpoint = torch.load(f"{output_path}/optimizer_state-{resume_iteration}.pt")
    if optimizer_type == "adam" or optimizer_type == "sgd":
        start_iteration = checkpoint["iteration"]
    else:
        start_iteration = checkpoint["lbfgs_iteration"]
        closure_count = checkpoint["iteration"]
    initial_vleft_x_model.load_state_dict(checkpoint['velocity_x_state_dict'])  # TODO: fix key
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    start_iteration = 0
    if optimizer_type == "lbfgs":
        closure_count = 0

# Get the current velocity model
initial_vleft_x = initial_vleft_x_model.current_params.to(device)

# Start optimization iteration
if optimizer_type == "adam" or optimizer_type == "sgd":
    for iteration in range(start_iteration, niteration):
        print(f"Iteration: {iteration} ---------------------------------------")
        t_start_opt = time.time()
        optimizer.zero_grad()  # Clear previous gradients

        # Make current initial velocity
        v0_field_grid = vel_fields_grid_true[0].clone()
        # replace left most node's vel-x to `initial_vleft_x`.
        v0_field_grid[0, 1:-1, 0] = initial_vleft_x[1:-1]
        # Map it back to the original velocity tensor following its ordering convention
        v0_field_flatten = v0_field_grid.permute(1, 0, 2).reshape(-1, 2)  # reshape to [lx*lx, dims]

        # Forward
        # Note: if we use nsteps=10, the resultant `prediction_velocities` will have the length of 1+10
        print("Forward...")
        vel_fields_flatten_pred = rollout_with_checkpointing(
            simulator=simulator,
            node_coords=node_coords,
            node_types=node_type,
            initial_velocities=v0_field_flatten,
            cells=cells,
            nsteps=inverse_timestep_range[1] - INPUT_SEQUENCE_LENGTH,
            checkpoint_interval=1,
            device=device)

        # Convert the velocity to velocity grid
        vel_fields_grid_pred = torch.reshape(
            vel_fields_flatten_pred, (len(vel_fields_flatten_pred), ly, lx, 2)).permute(0, 2, 1, 3)

        # Init visualizer
        vis = visualizer.VisMeshNet(
            mesh_type="quad",
            node_coords=node_coords.clone().detach().cpu().numpy(),
            node_type=node_type.clone().detach().cpu().numpy(),
            vel_true=velocities[:inverse_timestep_range[1]].clone().detach().cpu().numpy(),
            vel_pred=vel_fields_flatten_pred[:inverse_timestep_range[1]].clone().detach().cpu().numpy(),
            quad_grid_config=[lx, ly])

        # Get data to compare with target
        # data_vel_grid = vel_fields_grid_pred[
        #                 inverse_timestep_range[0]:inverse_timestep_range[1],  # timesteps
        #                 observation_patches[0][0]:observation_patches[0][1],  # x-range
        #                 observation_patches[1][0]:observation_patches[1][1]]  # y-range
        data_vel_fields_patches = tools.sample_patches(
            vel_fields_grid_pred,
            inverse_timestep_range[0], inverse_timestep_range[1],
            observation_patches_size[0], observation_patches_size[1],
            patch_locations=observation_patches_edge
        )

        # Loss
        mse_losses = [
            F.mse_loss(patch1, patch2)
            for patch1, patch2 in zip(target_vel_fields_patches, data_vel_fields_patches)]
        loss = torch.mean(torch.stack(mse_losses))
        print(f"MSE loss: {loss}")

        # Plot model: current velocity inference (before update)
        fig_model = vis.plot_model(
            timestep=0, title=f"Iteration={iteration}, MSE={loss.item():.3e}")
        fig_model.savefig(f"{output_path}/model-{iteration}.png")

        # Plot data: current velocity field (before update)
        # highlight_x = observation_patches[0][0] * x_conversion
        # highlight_y = observation_patches[1][0] * y_conversion
        # highlight_len_x = (observation_patches[0][1] - observation_patches[0][0]) * x_conversion
        # highlight_len_y = (observation_patches[1][1] - observation_patches[1][0]) * y_conversion
        highlight_regions = [
            (x_edge * x_conversion, y_edge * y_conversion, observation_patches_size[0] * x_conversion, observation_patches_size[1] * y_conversion)
            for x_edge, y_edge in observation_patches_edge]

        fig_data = vis.plot_field_compare(
            timestep=inverse_timestep_range[1]-INPUT_SEQUENCE_LENGTH,
            highlights=highlight_regions,
            title=f"Iteration={iteration}, MSE={loss.item():.3e}")
        fig_data.savefig(f"{output_path}/data_t{inverse_timestep_range[1]-INPUT_SEQUENCE_LENGTH}-{iteration}.png")

        # Backpropagation
        print("Backpropagate...")
        t_start_backprop = time.time()
        loss.backward()
        t_end_backprop = time.time()
        t_backprop = t_end_backprop - t_start_backprop

        # Update model
        print("Update parameters...")
        optimizer.step()

        # Record time for an optimization iteration
        t_end_opt = time.time()
        t_opt_iter = t_end_opt - t_start_opt

        # Save optimization status
        print("Save optimization status")
        if iteration % save_step == 0:
            torch.save({
                'iteration': iteration,
                't_opt_iter': t_opt_iter,
                't_backprop': t_backprop,
                'updated_velocities': v0_field_flatten.clone().detach().cpu().numpy(),
                'velocity_x_state_dict': tools.To_Torch_Model_Param(initial_vleft_x).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f"{output_path}/optimizer_state-{iteration}.pt")

