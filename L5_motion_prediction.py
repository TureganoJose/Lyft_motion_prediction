# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import math
# import packages
from IPython.display import display, clear_output
from IPython.core.display import display, HTML
import PIL
import gc
import zarr
import numpy as np
#import pandas as pd
from tqdm import tqdm
from typing import Dict
from collections import Counter
from prettytable import PrettyTable
import bisect
from L5_classes_experiments import CustomAgentDataset, CustomEgoDataset
import ssl


# level5 toolkit
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.filter import filter_agents_by_track_id, filter_agents_by_labels

# level5 toolkit
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset

# visualization
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from colorama import Fore, Back, Style
from scipy import signal

# deep learning
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision.models.resnet import resnet18, resnet50, resnet34

# Models
from backbone import ResNetBackbone
from mtp import MTP, MTPLoss





def get_scene(agent_id):
    frame_id = bisect.bisect_right(agents_ij[:, 0], agent_id) - 1
    scene_id = bisect.bisect_right(frames_ij[:, 0], frame_id) - 1

    scene = zarr_dataset.scenes[scene_id]
    frame = zarr_dataset.frames[frame_id]
    agent = zarr_dataset.agents[agent_id]
    return scene, (frame, frame_id), agent

class LabelEncoder:
    def __init__(self, max_size=500, default_val=-1):
        self.max_size = max_size
        self.labels = {}
        self.default_val = default_val

    @property
    def nlabels(self):
        return len(self.labels)

    def reset(self):
        self.labels = {}

    def partial_fit(self, keys):
        nlabels = self.nlabels
        available = self.max_size - nlabels

        if available < 1:
            return

        keys = set(keys)
        new_keys = list(keys - set(self.labels))

        if not len(new_keys):
            return

        self.labels.update(dict(zip(new_keys, range(nlabels, nlabels + available))))

    def fit(self, keys):
        self.reset()
        self.partial_fit(keys)

    def get(self, key):
        return self.labels.get(key, self.default_val)

    def transform(self, keys):
        return np.array(list(map(self.get, keys)))

    def fit_transform(self, keys, partial=True):
        self.partial_fit(keys) if partial else self.fit(keys)
        return self.transform(keys)


class CustomLyftDataset(Dataset):
    feature_mins = np.array([-17.336, -27.137, 0., 0., 0., -3.142, -37.833, -65.583],
                            dtype="float32")[None, None, None]

    feature_maxs = np.array([17.114, 20.787, 42.854, 42.138, 7.079, 3.142, 29.802, 35.722],
                            dtype="float32")[None, None, None]

    def __init__(self, zdataset, scenes=None, nframes=10, frame_stride=15, hbackward=10,
                 hforward=50, max_agents=150, agent_feature_dim=8):
        """
        Custom Lyft dataset reader.

        Parmeters:
        ----------
        zdataset: zarr dataset
            The root dataset, containing scenes, frames and agents

        nframes: int
            Number of frames per scene

        frame_stride: int
            The stride when reading the **nframes** frames from a scene

        hbackward: int
            Number of backward frames from  current frame

        hforward: int
            Number forward frames from current frame

        max_agents: int
            Max number of agents to read for each target frame. Note that,
            this also include the backward agents but not the forward ones.
        """
        super().__init__()
        self.zdataset = zdataset
        self.scenes = scenes if scenes is not None else []
        self.nframes = nframes
        self.frame_stride = frame_stride
        self.hbackward = hbackward
        self.hforward = hforward
        self.max_agents = max_agents

        self.nread_frames = (nframes - 1) * frame_stride + hbackward + hforward

        self.frame_fields = ['timestamp', 'agent_index_interval']

        self.agent_feature_dim = agent_feature_dim

        self.filter_scenes()

    def __len__(self):
        return len(self.scenes)

    def filter_scenes(self):
        self.scenes = [scene for scene in self.scenes if self.get_nframes(scene) > self.nread_frames]

    def __getitem__(self, index):
        return self.read_frames(scene=self.scenes[index])

    def get_nframes(self, scene, start=None):
        frame_start = scene["frame_index_interval"][0]
        frame_end = scene["frame_index_interval"][1]
        nframes = (frame_end - frame_start) if start is None else (frame_end - max(frame_start, start))
        return nframes

    def _read_frames(self, scene, start=None):
        nframes = self.get_nframes(scene, start=start)
        assert nframes >= self.nread_frames

        frame_start = scene["frame_index_interval"][0]

        start = start or frame_start + np.random.choice(nframes - self.nread_frames)
        frames = self.zdataset.frames.get_basic_selection(
            selection=slice(start, start + self.nread_frames),
            fields=self.frame_fields,
        )
        return frames

    def parse_frame(self, frame):
        return frame

    def parse_agent(self, agent):
        return agent

    def read_frames(self, scene, start=None, white_tracks=None, encoder=False):
        white_tracks = white_tracks or []
        frames = self._read_frames(scene=scene, start=start)

        agent_start = frames[0]["agent_index_interval"][0]
        agent_end = frames[-1]["agent_index_interval"][1]

        agents = self.zdataset.agents[agent_start:agent_end]

        X = np.zeros((self.nframes, self.max_agents, self.hbackward, self.agent_feature_dim), dtype=np.float32)
        target = np.zeros((self.nframes, self.max_agents, self.hforward, 2), dtype=np.float32)
        target_availability = np.zeros((self.nframes, self.max_agents, self.hforward), dtype=np.uint8)
        X_availability = np.zeros((self.nframes, self.max_agents, self.hbackward), dtype=np.uint8)

        for f in range(self.nframes):
            backward_frame_start = f * self.frame_stride
            forward_frame_start = f * self.frame_stride + self.hbackward
            backward_frames = frames[backward_frame_start:backward_frame_start + self.hbackward]
            forward_frames = frames[forward_frame_start:forward_frame_start + self.hforward]

            backward_agent_start = backward_frames[-1]["agent_index_interval"][0] - agent_start
            backward_agent_end = backward_frames[-1]["agent_index_interval"][1] - agent_start

            backward_agents = agents[backward_agent_start:backward_agent_end]

            le = LabelEncoder(max_size=self.max_agents)
            le.fit(white_tracks)
            le.partial_fit(backward_agents["track_id"])

            for iframe, frame in enumerate(backward_frames):
                backward_agent_start = frame["agent_index_interval"][0] - agent_start
                backward_agent_end = frame["agent_index_interval"][1] - agent_start

                backward_agents = agents[backward_agent_start:backward_agent_end]

                track_ids = le.transform(backward_agents["track_id"])
                mask = (track_ids != le.default_val)
                mask_agents = backward_agents[mask]
                mask_ids = track_ids[mask]
                X[f, mask_ids, iframe, :2] = mask_agents["centroid"]
                X[f, mask_ids, iframe, 2:5] = mask_agents["extent"]
                X[f, mask_ids, iframe, 5] = mask_agents["yaw"]
                X[f, mask_ids, iframe, 6:8] = mask_agents["velocity"]

                X_availability[f, mask_ids, iframe] = 1

            for iframe, frame in enumerate(forward_frames):
                forward_agent_start = frame["agent_index_interval"][0] - agent_start
                forward_agent_end = frame["agent_index_interval"][1] - agent_start

                forward_agents = agents[forward_agent_start:forward_agent_end]

                track_ids = le.transform(forward_agents["track_id"])
                mask = track_ids != le.default_val

                target[f, track_ids[mask], iframe] = forward_agents[mask]["centroid"]
                target_availability[f, track_ids[mask], iframe] = 1

        target -= X[:, :, [-1], :2]
        target *= target_availability[:, :, :, None]
        X[:, :, :, :2] -= X[:, :, [-1], :2]
        X *= X_availability[:, :, :, None]
        X -= self.feature_mins
        X /= (self.feature_maxs - self.feature_mins)

        if encoder:
            return X, target, target_availability, le
        return X, target, target_availability


# animation for scene
def animate_solution(images, timestamps=None):
    def animate(i):
        changed_artifacts = [im]
        im.set_data(images[i])
        if timestamps is not None:
            time_text.set_text(timestamps[i])
            changed_artifacts.append(im)
        return tuple(changed_artifacts)

    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    if timestamps is not None:
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=60, blit=True)

    # To prevent plotting image inline.
    plt.close()
    return anim

def visualize_rgb_image(dataset, index, title="", ax=None):
    """Visualizes Rasterizer's RGB image"""
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)

    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.imshow(im[::-1])

def create_animate_for_indexes(dataset, indexes):
    images = []
    timestamps = []

    for idx in indexes:
        data = dataset[idx]
        im = data["image"].transpose(1, 2, 0)
        im = dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
        clear_output(wait=True)
        images.append(PIL.Image.fromarray(im[::-1]))
        timestamps.append(data["timestamp"])

    anim = animate_solution(images, timestamps)
    return anim

def create_animate_for_scene(dataset, scene_idx):
    indexes = dataset.get_scene_indices(scene_idx)
    return create_animate_for_indexes(dataset, indexes)

if __name__ == '__main__':
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

    # get configuration yaml
    #cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")
    # --- Lyft configs ---
    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': 'resnet18',
            'history_num_frames': 10,
            'history_step_size': 1,
            'history_delta_time': 0.1,
            'future_num_frames': 50,
            'future_step_size': 1,
            'future_delta_time': 0.1
        },

        'raster_params': {
            'raster_size': [224, 224],
            'pixel_size': [0.5, 0.5],
            'ego_center': [0.25, 0.5],
            'map_type': 'py_semantic',
            'satellite_map_key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/aerial_map/aerial_map.png',
            'semantic_map_key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb',
            'dataset_meta_key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/meta.json',
            'filter_agents_threshold': 0.5
        },

        'train_data_loader': {
            'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr',
            'batch_size': 12,
            'shuffle': True,
            'num_workers': 0
        },

        'sample_data_loader': {
            'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr',
            'batch_size': 12,
            'shuffle': False,
            'num_workers': 16
        },

        'valid_data_loader': {
            'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr',
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4
        },

        'test_data_loader': {
            'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr',
            'batch_size': 8,
            'shuffle': False,
            'num_workers': 4
        },

        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,

            # 'eval_every_n_steps': -1
        }
    }

    dm = LocalDataManager()
    dataset_path = dm.require(cfg["sample_data_loader"]["key"])
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    ############################################################
    # Prepare all rasterizer and EgoDataset for each rasterizer
    ############################################################
    rasterizer_dict = {}
    dataset_dict = {}

    rasterizer_type_list = ["py_satellite", "satellite_debug", "py_semantic", "semantic_debug", "box_debug", "stub_debug"]
    cfg["raster_params"]["map_type"] = "py_semantic"
    semantic_rasterizer = build_rasterizer(cfg, dm)
    # Load agent dataset
    if dataset_path == 'C:\\Users\\jmartinez\\PycharmProjects\\L5_motion_prediction\\input\\lyft-motion-prediction-autonomous-vehicles\\scenes\\test.zarr':
        test_mask = np.load("C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")['arr_0']
        agent_dataset = CustomAgentDataset(cfg, zarr_dataset, semantic_rasterizer, agents_mask=test_mask)
    else:
        agent_dataset = CustomAgentDataset(cfg, zarr_dataset, semantic_rasterizer)


    # Number of agents
    n_cars = np.count_nonzero(agent_dataset.get_agent_labels() == 3)
    n_pedestrians = np.count_nonzero(agent_dataset.get_agent_labels() == 12)
    n_cyclists = np.count_nonzero(agent_dataset.get_agent_labels() == 14)

    # Agent indices (relative to agent dataset, not zarr_dataset)
    car_indices = list(np.nonzero(agent_dataset.get_agent_labels() == 3)[0])
    pedestrian_indices = list(np.nonzero(agent_dataset.get_agent_labels() == 12)[0])
    cyclists_indices = list(np.nonzero(agent_dataset.get_agent_labels() == 14)[0])

    # Subsets of agent dataset
    car_agent_dataset = Subset(agent_dataset,car_indices)
    pedestrian_agent_dataset = Subset(agent_dataset, pedestrian_indices)
    cyclist_agent_dataset = Subset(agent_dataset, cyclists_indices)

    # car_loader = DataLoader(car_agent_dataset)
    #
    # tr_it = iter(car_loader)
    # history_sizes = []
    # target_sizes = []
    # for itr in range(100):
    #     data = next(tr_it)
    #     history_sizes.append(torch.sum(data["history_availabilities"]))
    #     #target_sizes.append(torch.sum(data["target_availabilities"]))


    # # Load agent mask
    # agents_mask = agent_dataset.load_agents_mask()
    # past_mask = agents_mask[:, 0] >= 10
    # future_mask = agents_mask[:, 1] >= 1
    # agents_mask = past_mask * future_mask
    # agents_indices = np.nonzero(agents_mask)[0]

    # Ploting data

    # def moving_average(a, n=10) :
    #     ret = np.cumsum(a, dtype=float)
    #     ret[n:] = ret[n:] - ret[:-n]
    #     return ret[n - 1:] / n
    #
    # dt = 0.1
    # fs = 1/dt
    # fc = 2  # Cut-off frequency of the filter
    # w = fc / (fs / 2) # Normalize the frequency
    # b, a = signal.butter(5, w, 'low')
    #
    # random_agent_idx = np.random.randint(0, len(agent_dataset), size=10)
    # for i, agent_dataset_idx  in enumerate(random_agent_idx):
    #
    #     #agent_dataset_idx = 270
    #     data = agent_dataset[agent_dataset_idx]
    #
    #     # Plot rasterizer
    #     fig, axs = plt.subplots(2, 3)
    #     #visualize_rgb_image(agent_dataset, index=agent_dataset_idx, title="py_satellite", ax=axs[0, 0])
    #
    #     im = data["image"].transpose(1, 2, 0)
    #     im = agent_dataset.rasterizer.to_rgb(im)
    #     target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    #     draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    #     axs[0, 0].imshow(im[::-1])
    #
    #
    #     # Plot position of first agent
    #     pos_hist = data["history_positions"] + data["centroid"][:2]
    #     pos_fut = data["target_positions"] + data["centroid"][:2]
    #     axs[0, 1].scatter(pos_hist[:, 0], pos_hist[:, 1], color="blue")
    #     axs[0, 1].scatter(pos_fut[:, 0], pos_fut[:, 1], color="green")
    #
    #     # Calculate speed of car states
    #
    #     # Velocity
    #     pos = np.vstack((data["history_positions"][::-1], data["target_positions"]))
    #     pos = pos[~np.all(pos == 0, axis=1)] # Remove zeros
    #     ds = np.sqrt(np.sum(np.square(np.diff(pos, axis=0)), axis=1))
    #     velocity = 3.6 * ds/dt# km/h
    #     velocity_f = moving_average(velocity, 2) # moving average filter
    #     velocity_LPf = signal.filtfilt(b, a, velocity, axis=0)
    #     axs[1, 0].plot(velocity, color="blue")
    #     axs[1, 0].plot(velocity_f, color="red")
    #     axs[1, 0].plot(velocity_LPf, color="green")
    #     axs[1, 0].set_title('Speed at reference frame {:.2f}'.format(3.6*math.sqrt(data["velocity"][0]**2 + data["velocity"][1]**2)))
    #
    #     # Yaw rate
    #     yaw_hist = data["history_yaws"][::-1] * 180/math.pi
    #     yaw_fut = data["target_yaws"]* 180/ math.pi
    #     axs[1, 2].plot(yaw_hist, color="blue")
    #     axs[1, 2].plot(yaw_fut, color="red")
    #     axs[1, 2].set_title('yaw [deg]')
    #
    #     # Yaw rate
    #     yaw = np.vstack((data["history_yaws"][::-1], data["target_yaws"]))
    #     yaw = yaw[~np.all(yaw == 0, axis=1)] # Remove zeros
    #     dyaw = np.sqrt(np.square(np.diff(yaw, axis=0)))
    #     yaw_rate = dyaw/dt
    #     yaw_rate_f = moving_average(yaw_rate, 2)# moving average filter
    #     yaw_rate_LPf = signal.filtfilt(b, a, yaw_rate, axis=0)
    #     axs[1, 1].plot(yaw_rate, color="blue")
    #     axs[1, 1].plot(yaw_rate_f, color="red")
    #     axs[1, 1].plot(yaw_rate_LPf, color="green")
    #     axs[1, 1].set_title('yaw rate[rad/s]')
    #
    #
    #     # Acceleration
    #     dv = np.diff(velocity)
    #     acc = dv/dt# m/s2
    #     acc_f = moving_average(acc, 5) # moving average filter
    #     acc_LPf = signal.filtfilt(b, a, acc, axis=0)
    #     axs[0, 2].plot(acc, color="blue")
    #     axs[0, 2].plot(acc_f, color="red")
    #     axs[0, 2].plot(acc_LPf, color="green")
    #     axs[0, 2].set_title('accx [m/s]')
    #
    #     # Lateral speed
    #
    #     # Lateral acceleration
    #
    #     fig.show()



    # Neural network: MTP
    n_modes = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #backbone = ResNetBackbone('resnet18')
    ssl._create_default_https_context = ssl._create_unverified_context
    backbone = resnet18(pretrained=True)
    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    backbone.conv1 = nn.Conv2d(
        num_in_channels,
        backbone.conv1.out_channels,
        kernel_size=backbone.conv1.kernel_size,
        stride=backbone.conv1.stride,
        padding=backbone.conv1.padding,
        bias=False,
    )

    model = MTP(backbone, num_modes=n_modes, seconds=5, frequency_in_hz=10, input_shape=(num_in_channels, 224, 224))
    model = model.to(device)

    loss_function = MTPLoss(n_modes, 1, 5)

    current_loss = 10000

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    n_iter = 0

    minimum_loss = 0

    if n_modes == 2:
        # We expect to see 75% going_forward and
        # 25% going backward. So the minimum
        # classification loss is expected to be
        # 0.56234

        minimum_loss += 0.56234

    train_dataloader = DataLoader(agent_dataset, shuffle=cfg["train_data_loader"]["shuffle"], batch_size=cfg["train_data_loader"]["batch_size"],
                                 num_workers=cfg["train_data_loader"]["num_workers"])

    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []

    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        img = data["image"].to(device)

        #batch_velocity = np.sqrt( np.square(data["velocity"][:,0]) + np.square(data["velocity"][:, 1]))
        #agent_state_vector = np.hstack((batch_velocity, data["history_yaws"]*180/math.pi, data["extent"][:, 0]))
        agent_state_vector = data["car_states"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        ground_truth = data["target_positions"].to(device)
        #ground_truth = torch.reshape(ground_truth, (batch_size, 1, n_timesteps, 2) )

        optimizer.zero_grad()

        prediction = model(img, agent_state_vector)
        loss = loss_function(prediction, ground_truth, target_availabilities)
        loss.backward()
        optimizer.step()

        current_loss = loss.cpu().detach().numpy()

        print(f"Current loss is {current_loss:.4f}")
        if np.allclose(current_loss, minimum_loss, atol=1e-4):
            print(f"Achieved near-zero loss after {n_iter} iterations.")
            break

    torch.save(model.state_dict(), "C:/Users\jmartinez/PycharmProjects/L5_motion_prediction")

#
# def filter_by_track_id_and_scene(agents: np.ndarray, track_id: int, scene_idx: int):
#     track_id_idx = np.nonzero(agents["track_id"] == track_id)[0]
#     track_id_mask = np.zeros(len(agents), dtype=bool)
#     track_id_mask[track_id_idx]= True
#     scene = scenes[scene_idx]
#     frames_ij = scene["frame_index_interval"]
#     agents_idx = zarr_dataset.frames["agent_index_interval"][frames_ij[0]:frames_ij[1]]
#     agents_idx = np.arange(agents_idx[0, 0], agents_idx[-1][1])
#     scene_mask = np.zeros(len(agents), dtype=bool)
#     scene_mask[agents_idx] = True
#     return agents[np.logical_and(track_id_mask, scene_mask)]
#
# def get_mask_by_track_id_and_scene(agents: np.ndarray, track_id: int, scene_idx: int):
#     track_id_idx = np.nonzero(agents["track_id"] == track_id)[0]
#     track_id_mask = np.zeros(len(agents), dtype=bool)
#     track_id_mask[track_id_idx]= True
#     scene = scenes[scene_idx]
#     frames_ij = scene["frame_index_interval"]
#     agents_idx = zarr_dataset.frames["agent_index_interval"][frames_ij[0]:frames_ij[1]]
#     agents_idx = np.arange(agents_idx[0, 0], agents_idx[-1][1])
#     scene_mask = np.zeros(len(agents), dtype=bool)
#     scene_mask[agents_idx] = True
#     return np.logical_and(track_id_mask, scene_mask)
#
# def get_mask_by_track_id(agents: np.ndarray, track_id):
#     track_id_idx = np.nonzero(agents["track_id"] == track_id)[0]
#     track_id_mask = np.zeros(len(agents), dtype=bool)
#     track_id_mask[track_id_idx]= True
#     return track_id_mask
#
# def get_car_label(label_probabilities: np.ndarray, threshold: float) -> np.array:
#     return label_probabilities[:, 3] > threshold
#
# def get_mask_agents_by_car_labels(agents: np.ndarray, threshold: float = 0.5) -> np.ndarray:
#     label_indices = get_car_label(agents["label_probabilities"], threshold)
#     return label_indices
#
# def filter_agents_by_car_labels(agents: np.ndarray, threshold: float = 0.5) -> np.ndarray:
#     label_indices = get_car_label(agents["label_probabilities"], threshold)
#     return agents[label_indices]
#
# # Agents with track id 1 and scene idx 0
# # agents_id1 = filter_agents_by_track_id(agents[:-1], 1)
#
# # Available track_ids in scene
# scene_idx = 0
# frames_ij = scenes["frame_index_interval"][scene_idx]
# agents_ij = frames["agent_index_interval"][frames_ij[0]:frames_ij[1]]
# track_ids = np.unique(agents["track_id"][agents_ij[0, 0]:agents_ij[-1][1]])
#
# track_id = 1
#
# agents_id1 = filter_by_track_id_and_scene(agents[:], track_id, scene_idx)
# pos = agents_id1["centroid"]
# fig = plt.figure()
# scat = plt.scatter(pos[:, 0], pos[:, 1], cmap="RdYlGn")
# plt.show()
#
# # Plot only cars
# scene_idx = 0
# frames_ij = zarr_dataset.scenes["frame_index_interval"][scene_idx]
# agents_ij = zarr_dataset.frames["agent_index_interval"][frames_ij[0]:frames_ij[1]]
# track_ids = np.unique(agents["track_id"][agents_ij[0, 0]:agents_ij[-1][1]])
# agents_car_idx = get_mask_agents_by_car_labels(agents[:])
#
# for i_track_id in track_ids:
#     track_id = i_track_id
#     agents_id1_idx = get_mask_by_track_id_and_scene(agents[:], track_id, scene_idx)
#     pos = []
#     pos = agents["centroid"][np.logical_and(agents_id1_idx, agents_car_idx)]
#     if pos.shape[0]>10:
#         fig = plt.figure()
#         scat = plt.scatter(pos[:, 0], pos[:, 1], cmap="RdYlGn")
#         plt.show()
#
#
#
#
#
#
#
#
#
# mask = np.load("C:/Users\jmartinez/PycharmProjects/L5_motion_prediction/input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")['arr_0']
# agent_ids = np.where(mask)[0]
#
#
#
# def update_plot(i, frame_ij, scat):
#     i_frame = frame_ij[i]
#     frame = zarr_dataset.frames[i_frame]
#     agents_ij = frame["agent_index_interval"]
#     start_agent = agents_ij[0]
#     end_agent = agents_ij[1]
#     useful_agents = filter_agents_by_car_labels(agents[start_agent:end_agent]) #filter_agents_by_labels(agents[start_agent:end_agent])
#     pos = useful_agents["centroid"]
#     ids = useful_agents["track_id"]
#     #frame["timestamp"]
#     # Change the colors...
#     scat.set_array(ids)
#     # Change the x,y positions. This expects a _single_ 2xN, 2D array
#     scat.set_offsets(pos)
#
#
# # Plot one agent in one scene
# frames_ij = zarr_dataset.scenes["frame_index_interval"][0]
# agents_ij = zarr_dataset.frames["agent_index_interval"][frames_ij[0]:frames_ij[1]]
#
# last_agent = agents_ij.shape[0]
# track_id = np.unique(agents["track_id"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]])
#
# pos = agents["centroid"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]]
# ids = agents["track_id"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]]
# # plt.plot(pos[:, 0], pos[:, 1], 'o', color='black')
#
# fig = plt.figure()
# scat = plt.scatter(pos[:, 0], pos[:, 1], c=ids, cmap="RdYlGn")
#
# frame_range = np.arange(frames_ij[0], frames_ij[1])
# n_frames = frames_ij[1] - frames_ij[0]
# ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, fargs=(frame_range, scat),interval=10)
# #plt.show()
#
# # for itrack in track_id:
# #     agent_track_id_idx = np.where(agents["track_id"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]]==itrack)
# #     inters = np.intersect1d(agent_ids, agent_track_id_idx)
# #     pos = agents["centroid"][inters]
# #     if pos.shape[0] > 25:
# #         plt.plot(pos[:, 0], pos[:, 1],'o', color='black')
# #         plt.show()
#
#
# # plt.scatter(pos[1:1000, 0], pos[1:1000, 1], marker='.')
# # axes = plt.gca()
# # # axes.set_xlim([-2500, 1600])
# # # axes.set_ylim([-2500, 1600])
# # plt.title("ego_translation of frames")
#
# frames_ij = zarr_dataset.scenes["frame_index_interval"]
# agents_ij = zarr_dataset.frames["agent_index_interval"]
#
# scene = zarr_dataset.scenes[0]
# frames_ids = scene["frame_index_interval"]
# frames = zarr_dataset.frames[frames_ids[0]:frames_ids[1]]
# agents_ids = frames["agent_index_interval"]
#
#
#
# agent_id = agent_ids[np.random.choice(len(agent_ids))]
# # scene,(frame,frame_id), agent = get_scene(agent_id)
#
#
# #
# # HBACKWARD = 50
# # HFORWARD = 0
# # NFRAMES = 1
# # FRAME_STRIDE = 0
# # AGENT_FEATURE_DIM = 8
# # MAX_AGENTS = 150
# # agent_ids = np.where(test_mask)[0]
# # dt = CustomLyftDataset(
# #         zarr_dataset,
# #         nframes=NFRAMES,
# #         frame_stride=FRAME_STRIDE,
# #         hbackward=HBACKWARD,
# #         hforward=HFORWARD,
# #         max_agents=MAX_AGENTS,
# #         agent_feature_dim=AGENT_FEATURE_DIM,
# # )
# #
# # frames_ij = zarr_dataset.scenes["frame_index_interval"]
# # agents_ij = zarr_dataset.frames["agent_index_interval"]
# # scene, (frame,frame_id), agent = get_scene(agent_ids[39])
# # X, _, _, le = dt.read_frames(scene,
# #                              start=frame_id - HBACKWARD + 1,
# #                              white_tracks=[agent["track_id"]],
# #                              encoder=True
# #                              )
#
#
# ########################################################################
# ### Plot position of ego for all the frames
# # FRAME_DTYPE = [
# #     ("timestamp", np.int64),
# #     ("agent_index_interval", np.int64, (2,)),
# #     ("traffic_light_faces_index_interval", np.int64, (2,)),
# #     ("ego_translation", np.float64, (3,)),
# #     ("ego_rotation", np.float64, (3, 3)),
# # ]
# ########################################################################
#
# # frames = zarr_dataset.frames
#
# ## This is slow.
# # coords = np.zeros((len(frames), 2))
# # for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):
# #     frame = zarr_dataset.frames[idx_data]
# #     coords[idx_coord] = frame["ego_translation"][:2]
#
# # This is much faster!
# # coords = frames["ego_translation"][:, :2]
# #
# # plt.scatter(coords[:, 0], coords[:, 1], marker='.')
# # axes = plt.gca()
# # axes.set_xlim([-2500, 1600])
# # axes.set_ylim([-2500, 1600])
# # plt.title("ego_translation of frames")
#
#
#
#
# ########################################################################
# ### Counting types of agents and showing them in a pretty table
# # AGENT_DTYPE = [
# #     ("centroid", np.float64, (2,)),
# #     ("extent", np.float32, (3,)),
# #     ("yaw", np.float32),
# #     ("velocity", np.float32, (2,)),
# #     ("track_id", np.uint64),
# #     ("label_probabilities", np.float32, (len(LABELS),)),
# # ]
# ########################################################################
#
# agents = zarr_dataset.agents
# probabilities = agents["label_probabilities"]
# labels_indexes = np.argmax(probabilities, axis=1)
# counts = []
# for idx_label, label in enumerate(PERCEPTION_LABELS):
#     counts.append(np.sum(labels_indexes == idx_label))
#
# table = PrettyTable(field_names=["label", "counts"])
# for count, label in zip(counts, PERCEPTION_LABELS):
#     table.add_row([label, count])
# print(table)
#
# # Plot pos for agent #1
# agent_id = frames["agent_index_interval"][0, :]
# agent_idx = np.where(agents["track_id"] == agent_id[1])
# agent_pos = agents["centroid"][agent_idx[0][0], :]
# ego_pos = frames["ego_translation"][0, :2]
#
#
#
# cfg["raster_params"]["map_type"] = "py_satellite"
# rast = build_rasterizer(cfg, dm)
# dataset = EgoDataset(cfg, zarr_dataset, rast)
# data = dataset[80]
#
# im = data["image"].transpose(1, 2, 0)
# im = dataset.rasterizer.to_rgb(im)
# target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
# draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
# figsize = plt.subplots(figsize = (10,10))
# plt.title('Satellite View',fontsize=20)
# plt.imshow(im[::-1])
# plt.show()