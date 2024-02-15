import torch
from tqdm import tqdm
import numpy as np
import os
import sys
import torch_geometric.transforms as T
sys.path.append('/work2/08264/baagee/frontera/gns-main/')
from meshnet import learned_simulator
from meshnet.utils import datas_to_graph
from meshnet.utils import NodeType
from meshnet.transform_4face import MyFaceToEdge
transformer = T.Compose([MyFaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


INPUT_SEQUENCE_LENGTH = 1
dt = 0.01

def rollout_with_checkpointing(
        simulator: learned_simulator.MeshSimulator,
        node_coords,
        node_types,
        initial_velocities,
        cells,
        nsteps,
        device,
        checkpoint_interval=1,
        is_fixed_mesh=True
):

    # initial_velocities = velocities[:INPUT_SEQUENCE_LENGTH].squeeze().to(device)
    current_velocities = initial_velocities.squeeze()
    velocities_for_mask = initial_velocities.detach().clone().squeeze()

    predictions = []

    # Rollout
    for step in tqdm(range(nsteps), total=nsteps):

        # First, obtain data to form a graph
        current_node_coords = node_coords[0] if is_fixed_mesh else node_coords[step]
        current_node_type = node_types[0] if is_fixed_mesh else node_types[step]
        current_cell = cells[0] if is_fixed_mesh else cells[step]
        current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(
            torch.float32).contiguous().to(device)

        current_example = (
            (current_node_coords, current_node_type, current_velocities, current_cell, current_time_idx_vector),
            None)

        # Make graph
        graph = datas_to_graph(current_example, dt=dt, device=device)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        graph = transformer(graph).to(device)

        if step % checkpoint_interval == 0:
            predicted_next_velocity = torch.utils.checkpoint.checkpoint(
                simulator.predict_velocity,
                graph.x[:, 1:3],
                graph.x[:, 0],
                graph.edge_index,
                graph.edge_attr)
        else:
            predicted_next_velocity = simulator.predict_velocity(
                graph.x[:, 1:3],
                graph.x[:, 0],
                graph.edge_index,
                graph.edge_attr)

        # Get masks
        inflow_mask = (current_node_type == NodeType.INFLOW).clone().detach().squeeze()
        inflow_mask = inflow_mask.bool()[:, None].expand(-1, current_velocities.shape[-1])
        wall_mask = (current_node_type == NodeType.WALL_BOUNDARY).clone().detach().squeeze()
        wall_mask = wall_mask.bool()[:, None].expand(-1, current_velocities.shape[-1])

        # Maintain the current initial velocity on inflow nodes
        predicted_next_velocity = torch.where(
            inflow_mask, velocities_for_mask.squeeze(), predicted_next_velocity)
        # Replace predicted velocity on wall nodes to known values, which is 0.
        predicted_next_velocity = torch.where(
            wall_mask, velocities_for_mask.squeeze(), predicted_next_velocity)

        predictions.append(predicted_next_velocity)

        # Update current position for the next prediction
        current_velocities = predicted_next_velocity

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    return torch.cat((initial_velocities.unsqueeze(0), predictions))


