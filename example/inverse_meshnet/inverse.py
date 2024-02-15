import time
import sys
import os
import numpy as np
import json
import glob
import argparse
import torch.utils.checkpoint
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
ground_truth_npz = "porous_valid0.npz"
lx, ly = 250, 150
is_fixed_mesh = True

# Inputs for optimizer
optimizer_type = "adam"  # or lbfgs
niteration = 50
inverse_timestep_range = [8, 10]
inverse_node_range = [[10, 20], [0, 150]]
checkpoint_interval = 1
lr = 0.01
initial_vleft_x = tools.vel_autogen(
    ly=ly, shape_option="uniform", args={"peak": [0.15, 0.15], "npoints": ly})

# inputs for forward simulator
simulator_metadata = utils.read_metadata(data_path, "rollout", "metadata-10gnn.json")
model_file = "model-1400000.pt"
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

ds = data_loader.get_data_loader_by_trajectories(
    path=f"{data_path}/{ground_truth_npz}", fixed_mesh=is_fixed_mesh)
features = next(iter(ds))
node_coords = features[0].numpy()
velocities = features[2].numpy()

# # plot to check node order
# ntimesteps = len(features[2])
# node_grid = np.reshape(node_coords, (ntimesteps, ly, lx, 2)).transpose((0, 2, 1, 3))
# fig, ax = plt.subplots(figsize=(30, 20))
# ax.scatter(node_grid[0][:, :, 0], node_grid[0][:, :, 1], s=0.1)
# plt.show()

# transform velocity to velocity grid, and get the target velocity
ntimesteps = len(velocities)
# Note that node_coords follows column-based order. Therefore, we need to the following conversion
# TODO: confirm if the conversion is correct.
vel_grid = np.reshape(velocities, (ntimesteps, ly, lx, 2)).transpose((0, 2, 1, 3))
# Empty the initial vel-x that will be updated during optimization
vel_grid[0, 0, :, 0] = 0
v0 = torch.tensor(vel_grid[0]).to(device)
target_vel_grid = vel_grid[inverse_timestep_range[0]:inverse_timestep_range[1],  # timesteps
             inverse_node_range[0][0]:inverse_node_range[0][1],  # x-range
             inverse_node_range[1][0]:inverse_node_range[1][1], :]  # y-range
target_vel_grid_torch = torch.tensor(target_vel_grid).to(device)

# Initialize initial x-velocity
initial_vleft_x = torch.tensor(initial_vleft_x, requires_grad=True, device=device)
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

initial_vleft_x = initial_vleft_x_model.current_params.to(device)
# initial_vel_y = torch.full((ly, 1), 0).to(device)

# Start optimization iteration
if optimizer_type == "adam" or optimizer_type == "sgd":
    for iteration in range(start_iteration, niteration):
        print(f"Iteration: {iteration} ---------------------------------------")
        t_start_opt = time.time()
        optimizer.zero_grad()  # Clear previous gradients

        # Get ground truth velocity at the inversion timesteps and node coordinates
        ds = data_loader.get_data_loader_by_trajectories(
            path=f"{data_path}/{ground_truth_npz}", fixed_mesh=is_fixed_mesh)
        features = next(iter(ds))
        node_coords = features[0].to(device)  # (timesteps, nnode, ndims)
        node_type = features[1].to(device)  # (timesteps, nnode, )
        velocities = features[2].to(device)  # (timesteps, nnode, ndims)
        cells = features[3].to(device)

        # Make current initial velocity
        initial_vel_grid_torch = v0.clone()
        initial_vel_grid_torch[0, 1:-1, 0] = initial_vleft_x[1:-1]  # replace left most node's vel-x to `initial_vleft_x`.
        # Map it back to the original velocity tensor following its ordering convention
        initial_vel_flatten = initial_vel_grid_torch.permute(1, 0, 2).reshape(-1, 2)  # reshape to [lx*lx, dims]
        # TODO: Check if the `initial_vel_flatten` can recovoer `initial_vel_grid`

        # Forward
        # Note: if we use nsteps=10, the resultant `prediction_velocities` will have the length of 1+10
        print("Forward...")
        pred_vels = rollout_with_checkpointing(
            simulator=simulator,
            node_coords=node_coords,
            node_types=node_type,
            initial_velocities=initial_vel_flatten,
            cells=cells,
            nsteps=inverse_timestep_range[1] - INPUT_SEQUENCE_LENGTH,
            checkpoint_interval=1,
            device=device)

        # Convert the velocity to velocity grid
        pred_vels_grid = torch.reshape(pred_vels, (len(pred_vels), ly, lx, 2)).permute(0, 2, 1, 3)

        # Init visualizer
        vis = visualizer.VisMeshNet(
            mesh_type="quad",
            node_coords=node_coords.clone().detach().cpu().numpy(),
            node_type=node_type.clone().detach().cpu().numpy(),
            vel_true=velocities[:inverse_timestep_range[1]].clone().detach().cpu().numpy(),
            vel_pred=pred_vels[:inverse_timestep_range[1]].clone().detach().cpu().numpy(),
            quad_grid_config=[lx, ly])

        # Plot model: current velocity inference (before update)
        fig_model = vis.plot_model(timestep=0)
        fig_model.savefig(f"{output_path}/model-{iteration}.png")

        # Plot data: current velocity field (before update)
        fig_data = vis.plot_field_compare(timestep=inverse_timestep_range[1]-INPUT_SEQUENCE_LENGTH)
        fig_data.savefig(f"{output_path}/data_t{inverse_timestep_range[1]-INPUT_SEQUENCE_LENGTH}-{iteration}.png")

        t = 2
        fig_data = vis.plot_field_compare(timestep=t)
        fig_data.savefig(f"{output_path}/data_t{t}-{iteration}.png")

        # Get data to compare with target
        data_vel_grid = pred_vels_grid[inverse_timestep_range[0]:inverse_timestep_range[1],  # timesteps
                          inverse_node_range[0][0]:inverse_node_range[0][1],  # x-range
                          inverse_node_range[1][0]:inverse_node_range[1][1]]  # y-range

        # Loss
        loss = torch.mean((data_vel_grid - target_vel_grid_torch) ** 2)
        print(f"MSE loss: {loss}")

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
                'updated_velocities': initial_vel_flatten.clone().detach().cpu().numpy(),
                'velocity_x_state_dict': tools.To_Torch_Model_Param(initial_vleft_x).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f"{output_path}/optimizer_state-{iteration}.pt")

