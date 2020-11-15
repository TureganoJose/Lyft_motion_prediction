from typing import List, Optional, Tuple

import numpy as np

from l5kit.data import (
    filter_agents_by_labels,
    filter_tl_faces_by_frames,
    get_agents_slice_from_frames,
    get_tl_faces_slice_from_frames,
)
from l5kit.data.filter import filter_agents_by_frames, filter_agents_by_track_id
from l5kit.geometry import angular_distance, compute_agent_pose, rotation33_as_yaw, transform_point
from l5kit.kinematic import Perturbation
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from l5kit.sampling import get_future_slice, get_history_slice

import math

def custom_generate_agent_sample(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    render_context: RenderContext,
    history_num_frames: int,
    history_step_size: int,
    future_num_frames: int,
    future_step_size: int,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.
    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.
    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        raster_size (Tuple[int, int]): Desired output raster dimensions
        pixel_size (np.ndarray): Size of one pixel in the real world
        ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_frames (int): Amount of history frames to draw into the rasters
        history_step_size (int): Steps to take between frames, can be used to subsample history frames
        future_num_frames (int): Amount of history frames to draw into the rasters
        future_step_size (int): Steps to take between targets into the future
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
to train models that can recover from slight divergence from training set data
    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.
    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()  # this is the minimum slice of agents we need
    history_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    future_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent
        velocity_agent = agent["velocity"]
        label = np.nonzero(agent["label_probabilities"] > 0.5)[0]

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_coords_offset, future_yaws_offset, future_availability = _custom_create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
    )

    # history_num_frames + 1 because it also includes the current frame
    history_coords_offset, history_yaws_offset, history_availability = _custom_create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )


    # Extract track ids of all the neighbouring agents
    agents_track_ids = _custom_get_track_ids_from_frames(history_frames, history_agents)

    # Remove the ego agent (or reference agent)
    agents_track_ids = agents_track_ids[agents_track_ids!=selected_track_id]

    # getting the history from all the agents
    num_agents = agents_track_ids.shape[0]
    agents_coords_offset = np.empty([num_agents, history_num_frames + 1, 2])
    agents_velocity = np.empty([num_agents, history_num_frames + 1, 2])
    agents_availability = np.empty([num_agents, history_num_frames + 1, 1])
    agents_distance = np.empty([num_agents, history_num_frames + 1, 1])
    agents_state_vector = np.empty([num_agents, 7, history_num_frames + 1])

    for i, agent_track_id in enumerate(agents_track_ids):
        agent_coords_offset, agent_yaws_offset, agent_availability, agent_velocity, agent_distance, agent_state_vector = _custom_create_targets_for_lstm_encoding(
            history_num_frames + 1, history_frames, agent_track_id, history_agents, agent_from_world, agent_yaw_rad,
            history_coords_offset
        )
        agents_coords_offset[i, :, :] = agent_coords_offset
        agents_velocity[i, :, :] = agent_velocity
        agents_availability[i, :, 0] = agent_availability
        agents_distance[i, :, 0] = agent_distance.squeeze()
        agents_state_vector[i, :, :] = agent_state_vector

    #sorted_indeces = np.argsort(-np.count_nonzero(agents_coords_offset, axis=1)[:, 0]) # Sort by num of agents
    sorted_indeces = np.argsort(agents_distance[:, 0, 0])
    sorted_agents_coords_offsets = agents_coords_offset[sorted_indeces, :, :]
    sorted_agents_velocity_offsets = agents_velocity[sorted_indeces, :, :]
    sorted_agents_availability = agents_availability[sorted_indeces, :, :]
    sorted_agents_distance = agents_distance[sorted_indeces, :, :]
    sorted_agents_state_vector = agents_state_vector[sorted_indeces, :, :]
    sorted_agents_state_vector = sorted_agents_state_vector.reshape((num_agents*7, history_num_frames + 1))

    padded_agents_state_vector = np.zeros((100*7, history_num_frames + 1))
    if(sorted_agents_state_vector.shape[0]<100*7):
        padded_agents_state_vector[:sorted_agents_state_vector.shape[0], :sorted_agents_state_vector.shape[1]]=sorted_agents_state_vector
    else:
        padded_agents_state_vector[:700, :sorted_agents_state_vector.shape[1]] = sorted_agents_state_vector[:700, :]

    # full state vector of reference/ego agent
    agent_coords_offset, agent_yaws_offset, agent_availability, agent_velocity, agent_distance, ego_agent_state_vector = _custom_create_targets_for_lstm_encoding(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad,
        history_coords_offset)

    return {
        "image": input_im,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_availabilities": future_availability,
        "history_positions": history_coords_offset,
        #"history_all_agents_positions": sorted_agents_coords_offsets,
        "history_yaws": history_yaws_offset,
        "history_availabilities": history_availability,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
        "velocity": velocity_agent,
        #"velocity_all_agents": sorted_agents_velocity_offsets,
        #"availability_all_agents": sorted_agents_availability,
        #"distance_all_agents": sorted_agents_distance,
        "label": label,
        "num_agents": num_agents,
        "agents_state_vector": padded_agents_state_vector,
        "ego_agent_state_vector": ego_agent_state_vector
    }

def _custom_get_track_ids_from_frames(
        frames: np.ndarray,
        agents: List[np.ndarray]) -> np.array:
    agents_track_ids = []
    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        frame_agents = filter_agents_by_labels(frame_agents) # default threshold 0.5
        agents_track_ids.extend(frame_agents["track_id"])
    return np.unique(np.array(agents_track_ids))


def _custom_create_targets_for_deep_prediction(
    num_frames: int,
    frames: np.ndarray,
    selected_track_id: Optional[int],
    agents: List[np.ndarray],
    agent_from_world: np.ndarray,
    current_agent_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).
    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, availabilities
    """
    # How much the coordinates differ from the current state in meters.
    coords_offset = np.zeros((num_frames, 2), dtype=np.float32)
    yaws_offset = np.zeros((num_frames, 1), dtype=np.float32)
    availability = np.zeros((num_frames,), dtype=np.float32)

    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid = frame["ego_translation"][:2]
            agent_yaw = rotation33_as_yaw(frame["ego_rotation"])
        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(frame_agents, selected_track_id)[0]
                agent_centroid = agent["centroid"]
                agent_yaw = agent["yaw"]
            except IndexError:
                availability[i] = 0.0  # keep track of invalid futures/history
                continue

        coords_offset[i] = transform_point(agent_centroid, agent_from_world)
        yaws_offset[i] = angular_distance(agent_yaw, current_agent_yaw)
        availability[i] = 1.0
    return coords_offset, yaws_offset, availability

def _custom_create_targets_for_lstm_encoding(
    num_frames: int,
    frames: np.ndarray,
    selected_track_id: Optional[int],
    agents: List[np.ndarray],
    agent_from_world: np.ndarray,
    current_agent_yaw: float,
    ego_agent_coords_offset: np.ndarray # This is not necessarily the AV, just the reference agent
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray , np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).
    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, availabilities
    """
    # How much the coordinates differ from the current state in meters.
    coords_offset = np.zeros((num_frames, 2), dtype=np.float32)
    yaws_offset = np.zeros((num_frames, 1), dtype=np.float32)
    availability = np.zeros((num_frames,), dtype=np.float32)
    velocity = np.zeros((num_frames, 2), dtype=np.float32)
    distance_to_ego_agent = np.zeros((num_frames, 1), dtype=np.float32)
    agent_state_vector = np.zeros((7, num_frames), dtype=np.float32)

    for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid = frame["ego_translation"][:2]
            agent_yaw = rotation33_as_yaw(frame["ego_rotation"])
        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(frame_agents, selected_track_id)[0]
                agent_centroid = agent["centroid"]
                agent_yaw = agent["yaw"]
                agent_velocity = np.linalg.norm(agent["velocity"])
            except IndexError:
                availability[i] = 0.0  # keep track of invalid futures/history
                distance_to_ego_agent[i] = 500.0  # if it is not available, it is very far
                continue

        coords_offset[i] = transform_point(agent_centroid, agent_from_world)
        yaws_offset[i] = angular_distance(agent_yaw, current_agent_yaw)
        availability[i] = 1.0
        velocity[i] = np.array([agent_velocity * math.cos(yaws_offset[i]), agent_velocity * math.sin(yaws_offset[i])])
        distance_to_ego_agent[i] = np.linalg.norm(coords_offset[i] - ego_agent_coords_offset[i], axis=0)

        agent_state_vector = np.vstack( (coords_offset.T, yaws_offset.T,
                                         availability, velocity.T,  distance_to_ego_agent.T))
    return coords_offset, yaws_offset, availability, velocity, distance_to_ego_agent, agent_state_vector
