# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


print('l5kit imported')

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

# level5 toolkit
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager, filter_agents_by_labels

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

# deep learning
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34

# check files in directory
#print((os.listdir('../input/lyft-motion-prediction-autonomous-vehicles/')))

plt.rc('animation', html='jshtml')

#% matplotlib inline


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
    plt.show()
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

# Loading data
train = zarr.open("../lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr")
validation = zarr.open("../lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr")
test = zarr.open("../lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr/")
train.info

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "../lyft-motion-prediction-autonomous-vehicles"

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
        'num_workers': 4
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

print(cfg)


dm = LocalDataManager()
dataset_path = dm.require(cfg["sample_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


mask = np.load("C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")['arr_0']
agent_ids = np.where(mask)[0]



agents = zarr_dataset.agents
frames = zarr_dataset.frames
scenes = zarr_dataset.scenes

def get_car_label(label_probabilities: np.ndarray, threshold: float) -> np.array:
    return label_probabilities[:, 3] > threshold

def filter_agents_by_car_labels(agents: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    label_indices = get_car_label(agents["label_probabilities"], threshold)
    return agents[label_indices]


def update_plot(i, frame_ij, scat):
    i_frame = frame_ij[i]
    frame = zarr_dataset.frames[i_frame]
    agents_ij = frame["agent_index_interval"]
    start_agent = agents_ij[0]
    end_agent = agents_ij[1]
    useful_agents = filter_agents_by_car_labels(agents[start_agent:end_agent]) #filter_agents_by_labels(agents[start_agent:end_agent])
    pos = useful_agents["centroid"]
    ids = useful_agents["track_id"]
    #frame["timestamp"]
    # Change the colors...
    scat.set_array(ids)
    # Change the x,y positions. This expects a _single_ 2xN, 2D array
    scat.set_offsets(pos)




############################################################
# Prepare all rasterizer and EgoDataset for each rasterizer
############################################################
rasterizer_dict = {}
dataset_dict = {}

rasterizer_type_list = ["py_satellite", "satellite_debug", "py_semantic", "semantic_debug", "box_debug", "stub_debug"]
cfg["raster_params"]["map_type"] = rasterizer_type_list[0]
rast = build_rasterizer(cfg, dm)
agent_dataset = AgentDataset(cfg, zarr_dataset, rast)


# 1 Get similar trajectories from agents_class and raw data
# 2 Change coordinates from global to local
# 3 Get semantic pic


# Animation
scene_idx = 0
# Valid agent ids in scene
indexes = agent_dataset.get_scene_indices(scene_idx)
anim = create_animate_for_scene(agent_dataset, scene_idx)


# Plot one agent in one scene
frames_ij = zarr_dataset.scenes["frame_index_interval"][0]
agents_ij = zarr_dataset.frames["agent_index_interval"][frames_ij[0]:frames_ij[1]]

last_agent = agents_ij.shape[0]
track_id = np.unique(agents["track_id"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]])

pos = agents["centroid"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]]
ids = agents["track_id"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]]
# plt.plot(pos[:, 0], pos[:, 1], 'o', color='black')

fig = plt.figure()
scat = plt.scatter(pos[:, 0], pos[:, 1], c=ids, cmap="RdYlGn")

frame_range = np.arange(frames_ij[0], frames_ij[1])
n_frames = frames_ij[1] - frames_ij[0]
ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, fargs=(frame_range, scat),interval=10)
# plt.show()

















# for itrack in track_id:
#     agent_track_id_idx = np.where(agents["track_id"][agents_ij[0, 0]:agents_ij[last_agent-1, 1]]==itrack)
#     inters = np.intersect1d(agent_ids, agent_track_id_idx)
#     pos = agents["centroid"][inters]
#     if pos.shape[0] > 25:
#         plt.plot(pos[:, 0], pos[:, 1],'o', color='black')
#         plt.show()


# plt.scatter(pos[1:1000, 0], pos[1:1000, 1], marker='.')
# axes = plt.gca()
# # axes.set_xlim([-2500, 1600])
# # axes.set_ylim([-2500, 1600])
# plt.title("ego_translation of frames")

frames_ij = zarr_dataset.scenes["frame_index_interval"]
agents_ij = zarr_dataset.frames["agent_index_interval"]

scene = zarr_dataset.scenes[0]
frames_ids = scene["frame_index_interval"]
frames = zarr_dataset.frames[frames_ids[0]:frames_ids[1]]
agents_ids = frames["agent_index_interval"]



agent_id = agent_ids[np.random.choice(len(agent_ids))]
scene,(frame,frame_id), agent = get_scene(agent_id)
scene,(frame,frame_id), agent["track_id"]

HBACKWARD = 15
HFORWARD = 0
NFRAMES = 1
FRAME_STRIDE = 0
AGENT_FEATURE_DIM = 8
MAX_AGENTS = 150

dt = CustomLyftDataset(
        zarr_dataset,
        nframes=NFRAMES,
        frame_stride=FRAME_STRIDE,
        hbackward=HBACKWARD,
        hforward=HFORWARD,
        max_agents=MAX_AGENTS,
        agent_feature_dim=AGENT_FEATURE_DIM,
)
dt.nread_frames
########################################################################
### Plot position of ego for all the frames
# FRAME_DTYPE = [
#     ("timestamp", np.int64),
#     ("agent_index_interval", np.int64, (2,)),
#     ("traffic_light_faces_index_interval", np.int64, (2,)),
#     ("ego_translation", np.float64, (3,)),
#     ("ego_rotation", np.float64, (3, 3)),
# ]
########################################################################

########################################################################
### Counting types of agents and showing them in a pretty table
# AGENT_DTYPE = [
#     ("centroid", np.float64, (2,)),
#     ("extent", np.float32, (3,)),
#     ("yaw", np.float32),
#     ("velocity", np.float32, (2,)),
#     ("track_id", np.uint64),
#     ("label_probabilities", np.float32, (len(LABELS),)),
# ]
########################################################################

agents = zarr_dataset.agents
probabilities = agents["label_probabilities"]
labels_indexes = np.argmax(probabilities, axis=1)
counts = []
for idx_label, label in enumerate(PERCEPTION_LABELS):
    counts.append(np.sum(labels_indexes == idx_label))

table = PrettyTable(field_names=["label", "counts"])
for count, label in zip(counts, PERCEPTION_LABELS):
    table.add_row([label, count])
print(table)










cfg["raster_params"]["map_type"] = "py_satellite"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
data = dataset[80]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
figsize = plt.subplots(figsize = (10,10))
plt.title('Satellite View',fontsize=20)
plt.imshow(im[::-1])
plt.show()