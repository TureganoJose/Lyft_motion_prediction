import bisect
import warnings
from functools import partial
from typing import Optional, Tuple, cast
from pathlib import Path
from zarr import convenience

import numpy as np
from torch.utils.data import Dataset

from l5kit.data import (
    ChunkedDataset,
    get_agents_slice_from_frames,
    get_frames_slice_from_scenes,
    get_tl_faces_slice_from_frames,
)
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.sampling import generate_agent_sample

from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE
from classes.custom_agent_sampling import custom_generate_agent_sample

# WARNING: changing these values impact the number of instances selected for both train and inference!
MIN_FRAME_HISTORY = 10  # minimum number of frames an agents must have in the past to be picked
MIN_FRAME_FUTURE = 10  # minimum number of frames an agents must have in the future to be picked


class CustomEgoDataset(Dataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN
        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not)
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
None if not desired
        """
        self.perturbation = perturbation
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.rasterizer = rasterizer

        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]

        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
        )
        # build a partial so we don't have to access cfg each time
        self.custom_sample_function = partial(
            custom_generate_agent_sample,
            render_context=render_context,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
        )

    def __len__(self) -> int:
        """
        Get the number of available AV frames
        Returns:
            int: the number of elements in the dataset
        """
        return len(self.dataset.frames)

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame
        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp
        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        try:
            if self.cfg["raster_params"]["disable_traffic_light_faces"]:
                tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
        except KeyError:
            warnings.warn(
                "disable_traffic_light_faces not found in config, this will raise an error in the future",
                RuntimeWarning,
                stacklevel=2,
            )
        data = self.custom_sample_function(state_index, frames, self.dataset.agents, self.dataset.tl_faces, track_id)
        # 0,1,C -> C,0,1
        image = data["image"].transpose(2, 0, 1)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        # Additional car states
        n_hist = np.sum(data["history_availabilities"])
        dt = 0.1

        speed = np.sqrt(data["velocity"][0] ** 2 + data["velocity"][1] ** 2)
        yaw_hist_mean = np.sum(history_yaws) / n_hist
        pos = history_positions[::-1]
        ds = np.sqrt(np.sum(np.square(np.diff(pos, axis=0)), axis=1))
        velocity = 3.6 * ds / dt  # km/h
        acc_avg = (velocity[-1] - velocity[0]) / (n_hist * dt)
        car_states = np.array([speed, yaw_hist_mean, acc_avg, data["extent"][0]], dtype=np.float32)

        label = data["label"]

        return {
            "image": image,
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "history_positions": history_positions,
            "history_all_agents_positions": data["history_all_agents_positions"],
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            "world_to_image": data["raster_from_world"],  # TODO deprecate
            "raster_from_world": data["raster_from_world"],
            "raster_from_agent": data["raster_from_agent"],
            "agent_from_world": data["agent_from_world"],
            "world_from_agent": data["world_from_agent"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
            "velocity": data["velocity"],
            "agents_velocity": data["velocity_all_agents"],
            "label": label,
            "car_states": car_states,
            "num_agents": data["num_agents"]
        }

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element
        Args:
            index (int): index of the element to retrieve
        Returns: please look get_frame signature and docstring
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index)

    def get_scene_dataset(self, scene_index: int) -> "CustomEgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.
        Args:
            scene_index (int): the scene index of the new dataset
        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data
        """
        # copy everything to avoid references (scene is already detached from zarr if get_combined_scene was called)
        scenes = self.dataset.scenes[scene_index: scene_index + 1].copy()
        frame_slice = get_frames_slice_from_scenes(*scenes)
        frames = self.dataset.frames[frame_slice].copy()
        agent_slice = get_agents_slice_from_frames(*frames[[0, -1]])
        tl_slice = get_tl_faces_slice_from_frames(*frames[[0, -1]])

        agents = self.dataset.agents[agent_slice].copy()
        tl_faces = self.dataset.tl_faces[tl_slice].copy()

        frames["agent_index_interval"] -= agent_slice.start
        frames["traffic_light_faces_index_interval"] -= tl_slice.start
        scenes["frame_index_interval"] -= frame_slice.start

        dataset = ChunkedDataset("")
        dataset.agents = agents
        dataset.tl_faces = tl_faces
        dataset.frames = frames
        dataset.scenes = scenes

        return CustomEgoDataset(self.cfg, dataset, self.rasterizer, self.perturbation)

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. EgoDataset iterates over frames, so this is just a matter
        of finding the scene boundaries.
        Args:
            scene_idx (int): index of the scene
        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        return np.arange(*scenes[scene_idx]["frame_index_interval"])

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. EgoDataset iterates over frames, so this will be a single element
        Args:
            frame_idx (int): index of the scene
        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        frames = self.dataset.frames
        assert frame_idx < len(frames), f"frame_idx {frame_idx} is over len {len(frames)}"
        return np.asarray((frame_idx,), dtype=np.int64)

    def __str__(self) -> str:
        return self.dataset.__str__()


class CustomAgentDataset(CustomEgoDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            agents_mask: Optional[np.ndarray] = None,
            min_frame_history: int = MIN_FRAME_HISTORY,
            min_frame_future: int = MIN_FRAME_FUTURE,
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

        super(CustomAgentDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation)
        if agents_mask is None:  # if not provided try to load it from the zarr
            agents_mask = self.load_agents_mask()
            past_mask = agents_mask[:, 0] >= min_frame_history
            future_mask = agents_mask[:, 1] >= min_frame_future
            agents_mask = past_mask * future_mask

            if min_frame_history != MIN_FRAME_HISTORY:
                warnings.warn(
                    f"you're running with custom min_frame_history of {min_frame_history}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if min_frame_future != MIN_FRAME_FUTURE:
                warnings.warn(
                    f"you're running with custom min_frame_future of {min_frame_future}", RuntimeWarning, stacklevel=2
                )
        else:
            warnings.warn("you're running with a custom agents_mask", RuntimeWarning, stacklevel=2)

        # store the valid agents indexes
        self.agents_indices = np.nonzero(agents_mask)[0]
        # this will be used to get the frame idx from the agent idx
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
        self.agents_mask = agents_mask

        # store each agent category

    def load_agents_mask(self) -> np.ndarray:
        """
        Loads a boolean mask of the agent availability stored into the zarr. Performs some sanity check against cfg.
        Returns: a boolean mask of the same length of the dataset agents
        """
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]

        agents_mask_path = Path(self.dataset.path) / f"agents_mask/{agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            warnings.warn(
                f"cannot find the right config in {self.dataset.path},\n"
                f"your cfg has loaded filter_agents_threshold={agent_prob};\n"
                "but that value doesn't have a match among the agents_mask in the zarr\n"
                "Mask will now be generated for that parameter.",
                RuntimeWarning,
                stacklevel=2,
            )
            select_agents(
                self.dataset,
                agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )

        agents_mask = convenience.load(str(agents_mask_path))  # note (lberg): this doesn't update root
        return agents_mask

    def __len__(self) -> int:
        """
        length of the available and reliable agents (filtered using the mask)
        Returns: the length of the dataset
        """
        return len(self.agents_indices)

    def __getitem__(self, index: int) -> dict:
        """
        Differs from parent by iterating on agents and not AV.
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        index = self.agents_indices[index]  # bad practice here
        track_id = self.dataset.agents[index]["track_id"]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)

        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index, track_id=track_id)  # .update({'label': agent_label})

    def get_scene_dataset(self, scene_index: int) -> "CustomAgentDataset":
        """
        Differs from parent only in the return type.
        Instead of doing everything from scratch, we rely on super call and fix the agents_mask
        """

        new_dataset = super(CustomAgentDataset, self).get_scene_dataset(scene_index).dataset

        # filter agents_bool values
        frame_interval = self.dataset.scenes[scene_index]["frame_index_interval"]
        # ASSUMPTION: all agents_index are consecutive
        start_index = self.dataset.frames[frame_interval[0]]["agent_index_interval"][0]
        end_index = self.dataset.frames[frame_interval[1] - 1]["agent_index_interval"][1]
        agents_mask = self.agents_mask[start_index:end_index].copy()

        return CustomAgentDataset(
            self.cfg, new_dataset, self.rasterizer, self.perturbation, agents_mask  # overwrite the loaded one
        )

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. Here __getitem__ iterate over valid agents indices.
        This means ``__getitem__(0)`` matches the first valid agent in the dataset.
        Args:
            scene_idx (int): index of the scene
        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        frame_slice = get_frames_slice_from_scenes(scenes[scene_idx])
        agent_slice = get_agents_slice_from_frames(*self.dataset.frames[frame_slice][[0, -1]])

        mask_valid_indices = (self.agents_indices >= agent_slice.start) * (self.agents_indices < agent_slice.stop)
        indices = np.nonzero(mask_valid_indices)[0]
        return indices

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. Here __getitem__ iterate over valid agents indices.
        This means ``__getitem__(0)`` matches the first valid agent in the dataset.
        Args:
            frame_idx (int): index of the scene
        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """

        assert frame_idx < len(self.dataset.frames), f"frame_idx {frame_idx} is over len {len(self.dataset.frames)}"

        # avoid accessing zarr here as we already have the information in `cumulative_sizes_agents`
        agent_start = self.cumulative_sizes_agents[frame_idx - 1] if frame_idx > 0 else 0
        agent_end = self.cumulative_sizes_agents[frame_idx]

        mask_valid_indices = (self.agents_indices >= agent_start) * (self.agents_indices < agent_end)
        indices = np.nonzero(mask_valid_indices)[0]
        return indices