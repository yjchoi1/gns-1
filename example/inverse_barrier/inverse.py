import time
import os
import numpy as np
import toml
import json
import glob
import torch.utils.checkpoint

from example.inverse_barrier.forward import rollout_with_checkpointing
from example.inverse_problem.utils import To_Torch_Model_Param
from example.inverse_barrier.utils import *

from gns import reading_utils
from gns import train

path = "data/"

# Inputs for optimization
nepoch = 10
checkpoint_interval = 1
lr = 1.0
# initial location guess of barriers
barrier_locations = [[0.9, 0.2], [0.9, 0.7]]  # x and z at lower edge
barrier_info = {
    "barrier_height": 0.2,
    "barrier_width": 0.1,
    "base_height": 0.1  # lower boundary of the simulation domain
}
search_area = [[0.75, 0.15], [0.1, 0.3], [0.1, 0.9]]
n_farthest_particles = 100

# inputs for ground truth
ground_truth_npz = "trajectory0.npz"
ground_truth_mpm_inputfile = "mpm_input.json"

# Inputs for forward simulator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
dt_mpm = 0.0025
model_path = "data/"
model_file = "model-5600000.pt"
simulator_metadata_path = "data/"
simulator_metadata_file = "gns_metadata.json"

# Resume options
resume = False
resume_epoch = 1

# Save options
output_dir = "data/outputs/"
save_step = 1
# Set output folder
if not os.path.exists(f"{output_dir}"):
    os.makedirs(f"{output_dir}")

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path, "rollout", file_name=simulator_metadata_file)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth barrier positions
f = open(f"{path}/{ground_truth_mpm_inputfile}")
mpm_inputs = json.load(f)
barrier_geometries_true = mpm_inputs["gen_cube_from_data"]["sim_inputs"][0]["obstacles"]["cubes"]
barrier_locs_true = []
for geometry in barrier_geometries_true:
    loc = [geometry[0], geometry[2]]
    barrier_locs_true.append(loc)

# Get particle positions
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
positions = torch.tensor(mpm_trajectory[0][0])
particle_types = torch.tensor(mpm_trajectory[0][1])

# Get positions by particle type
kinematic_positions, stationary_positions = get_positions_by_type(
    positions, particle_types)

# Generate barrier mass filled with material points
dist_between_particles = mpm_inputs["domain_size"]/mpm_inputs["sim_resolution"][0]/2
barrier_particles = fill_cuboid_with_particles(
    len_x=barrier_info["barrier_width"],
    len_y=barrier_info["barrier_height"],
    len_z=barrier_info["barrier_width"],
    spacing=dist_between_particles).to(device)

# Get ground truth particle positions at the last timestep for `n_farthest_particles` to compute loss
runout_end_true = get_runout_end(
    kinematic_positions[-1], n_farthest_particles).to(device)

# Initialize barrier locations
barrier_locs_torch = torch.tensor(
    barrier_locations, requires_grad=True, device=device)
barrier_locs_param = To_Torch_Model_Param(barrier_locs_torch)

# Set up the optimizer
optimizer = torch.optim.LBFGS(barrier_locs_param.parameters(), lr=lr, history_size=100)

# Resume
if resume:
    print(f"Resume from the previous state: epoch{resume_epoch}")
    checkpoint = torch.load(f"{output_dir}/optimizer_state-{resume_epoch}.pt")
    start_epoch = checkpoint["epoch"]
    barrier_locs_param.load_state_dict(checkpoint['updated_barrier_loc_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    start_epoch = 0
barrier_locs = barrier_locs_param.current_params

# Start optimization iteration
for epoch in range(start_epoch, nepoch):
    print(f"Epoch outside closure: {epoch}")
    start = time.time()

    # Variables to that saves variables in closure during optimization
    values_for_save = {}  # variables to save as outputs
    values_for_vis = {"pred": {}, "true": {}}   # variables just for visualize current optimization state

    def closure():
        print(f"Epoch in closure: {epoch}")
        optimizer.zero_grad()  # Clear previous gradients

        # Make current barrier particles with the current locations
        base_height = torch.tensor(barrier_info["base_height"])
        current_barrier_particles = get_barrier_particles(
            barrier_particles, barrier_locs, base_height)

        # Make X0 with current barrier particles
        current_initial_positions, current_particle_type, current_n_particles_per_example = get_features(
            kinematic_positions, current_barrier_particles, device)

        # GNS rollout
        predicted_positions = rollout_with_checkpointing(
            simulator=simulator,
            initial_positions=current_initial_positions,
            particle_types=current_particle_type,
            n_particles_per_example=current_n_particles_per_example,
            nsteps=250,
            checkpoint_interval=checkpoint_interval,
            knwon_positions=current_initial_positions[:, 0, :]
        )

        # Get predicted position at the last timestep
        kinematic_positions_pred, stationary_positions_pred = get_positions_by_type(
            predicted_positions, current_particle_type)
        runout_end_pred = get_runout_end(
            kinematic_positions_pred[-1], n_farthest_particles)

        # Compute loss with before update
        # TODO (yc): regularization and constraint for loss
        loss = torch.mean((runout_end_pred - runout_end_true)**2)
        print(f"True barrier locations: {barrier_locs_true}")
        print(f"Epoch {epoch}:")
        print(f"loss {loss.item():.8f}")

        # Save necessary variables to visualize current optimization state
        values_for_vis["pred"]["kinematic_positions"] = kinematic_positions_pred  # torch.tensor
        values_for_vis["pred"]["runout_end"] = runout_end_pred  # torch.tensor
        values_for_vis["pred"]["barrier_locs"] = barrier_locs  # torch.tensor
        values_for_vis["true"]["kinematic_positions"] = kinematic_positions  # torch.tensor
        values_for_vis["true"]["runout_end"] = runout_end_true  # torch.tensor
        values_for_vis["true"]["barrier_locs"] = torch.tensor(barrier_locs_true)  # torch.tensor

        # Vis
        visualize_state(
            vis_data=values_for_vis,
            barrier_info=barrier_info,
            mpm_inputs=mpm_inputs,
            write_path=f"{output_dir}/status-{epoch}.png")
        # Animation
        if epoch % save_step == 0:
            render_animation(
                predicted_positions,
                current_particle_type,
                mpm_inputs,
                timestep_stride=10,
                write_path=f"{output_dir}/trj-{epoch}.gif")

        # Save necessary variables to save as output
        values_for_save["current_barrier_loc"] = barrier_locs.clone().detach().cpu().numpy()
        values_for_save["predicted_positions"] = predicted_positions.clone().detach().cpu().numpy()
        values_for_save["particle_type"] = current_particle_type.clone().detach().cpu().numpy()

        # Update barrier locations
        print("Backpropagate...")
        loss.backward()
        # Print updated barrier locations
        print(f"Updated barrier locations: {barrier_locs.detach().cpu().numpy()}")

        # Save updated state
        values_for_save["updated_barrier_loc"] = barrier_locs.clone().detach().cpu().numpy()

        return loss

    # Perform optimization step
    loss = optimizer.step(closure)

    # Measure time for an iteration
    end = time.time()
    time_for_iteration = end - start

    # Save optimizer state
    torch.save({
        'epoch': epoch,
        'loss': loss.item(),
        'time_spent': time_for_iteration,
        'save_values': values_for_save,
        'updated_barrier_loc_state_dict': To_Torch_Model_Param(barrier_locs).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{output_dir}/optimizer_state-{epoch}.pt")





    #
    # # TODO (yc): utils.visualizer
    #
    # # Perform optimization step
    # optimizer.step()



    # Save and report optimization status






