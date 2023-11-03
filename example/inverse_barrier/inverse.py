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
import optimizations

from gns import reading_utils
from gns import train

path = "data/"

# Inputs for optimization
optimizer_type = "adam"  # adam or lbfgs
nepoch = 15
lr = 0.01  # use 0.5 - 1.0 for lbfgs and 0.01 or smaller to adam
# initial location guess of barriers
barrier_locations = [[1.0, 0.20], [0.9, 0.7]]  # x and z at lower edge
# prescribed constraints for barrier geometry
barrier_info = {
    "barrier_height": 0.2,
    "barrier_width": 0.1,
    "base_height": 0.1,  # lower boundary of the simulation domain
    "search_area": [[0.7, 1.0], [0.2, 0.7]]
}
n_farthest_particles = 100

# inputs for ground truth
ground_truth_npz = "trajectory0.npz"
ground_truth_mpm_inputfile = "mpm_input.json"

# Inputs for forward simulator
nsteps = 250
checkpoint_interval = 1
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
save_step = 10
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
if optimizer_type == "lbfgs":
    optimizer = torch.optim.LBFGS(barrier_locs_param.parameters(), lr=lr, history_size=100)
elif optimizer_type == "adam":
    optimizer = torch.optim.Adam(barrier_locs_param.parameters(), lr=lr)
else:
    raise ValueError("Check `optimizer_type`")

# Resume TODO (yc): depends on optimizer type
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

    if optimizer_type == "lbfgs":
        barrier_locs, _ = optimizations.lbfgs(
            simulator,
            nsteps,
            mpm_inputs,
            epoch,
            checkpoint_interval,
            runout_end_true,
            n_farthest_particles,
            kinematic_positions,
            barrier_info,
            barrier_particles,
            barrier_locs,
            barrier_locs_true,
            optimizer,
            output_dir,
            device)

    elif optimizer_type == "adam":
        barrier_locs, _ = optimizations.adam(
            simulator,
            nsteps,
            mpm_inputs,
            epoch,
            checkpoint_interval,
            runout_end_true,
            n_farthest_particles,
            kinematic_positions,
            barrier_info,
            barrier_particles,
            barrier_locs,
            barrier_locs_true,
            optimizer,
            output_dir,
            device)

    else:
        raise ValueError("Check `optimizer type`")


    #
    # # Save animation after epoch
    # if epoch % save_step == 0:
    #     render_animation(
    #         predicted_positions,
    #         current_particle_type,
    #         mpm_inputs,
    #         timestep_stride=10,
    #         write_path=f"{output_dir}/trj-{epoch}.gif")






