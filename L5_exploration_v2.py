import keyboard

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
from numpy import linalg as LA

#import pandas as pd
from tqdm import tqdm
from typing import Dict
from collections import Counter
from prettytable import PrettyTable
import bisect
#from L5_classes_experiments import CustomAgentDataset, CustomEgoDataset
from classes.custom_classes import CustomAgentDataset, CustomEgoDataset
import ssl


# level5 toolkit
import l5kit
print(l5kit.__version__)
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.filter import filter_agents_by_track_id, filter_agents_by_labels

# level5 toolkit
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR

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
            'filter_agents_threshold': 0.5,
            'disable_traffic_light_faces': False
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


    # # Number of agents
    # n_cars = np.count_nonzero(agent_dataset.get_agent_labels() == 3)
    # n_pedestrians = np.count_nonzero(agent_dataset.get_agent_labels() == 12)
    # n_cyclists = np.count_nonzero(agent_dataset.get_agent_labels() == 14)
    #
    # # Agent indices (relative to agent dataset, not zarr_dataset)
    # car_indices = list(np.nonzero(agent_dataset.get_agent_labels() == 3)[0])
    # pedestrian_indices = list(np.nonzero(agent_dataset.get_agent_labels() == 12)[0])
    # cyclists_indices = list(np.nonzero(agent_dataset.get_agent_labels() == 14)[0])
    #
    # # Subsets of agent dataset
    # car_agent_dataset = Subset(agent_dataset,car_indices)
    # pedestrian_agent_dataset = Subset(agent_dataset, pedestrian_indices)
    # cyclist_agent_dataset = Subset(agent_dataset, cyclists_indices)
    #
    # car_loader = DataLoader(car_agent_dataset)
    #
    # tr_it = iter(car_loader)
    # history_sizes = []
    # target_sizes = []
    # for itr in range(100):
    #     data = next(tr_it)
    #     history_sizes.append(torch.sum(data["history_availabilities"]))
    #     #target_sizes.append(torch.sum(data["target_availabilities"]))


    # Load agent mask
    agents_mask = agent_dataset.load_agents_mask()
    past_mask = agents_mask[:, 0] >= 10
    future_mask = agents_mask[:, 1] >= 1
    agents_mask = past_mask * future_mask
    agents_indices = np.nonzero(agents_mask)[0]

    # Plotting data
    plt.figure(0)
    all_speeds = LA.norm(zarr_dataset.agents['velocity'][agent_dataset.agents_indices], axis=1)*3.6
    plt.hist(all_speeds, density=True, bins=80)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Speed km/h')

    ## Plotting availability for a sample of 1000 agents
    #plt.figure(1)
    #random_agent_idx = np.random.randint(0, len(agent_dataset), size=10000)
    #availabilities = []
    #for i, agent_dataset_idx in enumerate(random_agent_idx):
    #    data = agent_dataset[agent_dataset_idx]
    #    availabilities.append(np.sum(data["history_availabilities"]))
    #plt.hist(availabilities, density=True, bins=11)  # `density=False` would make counts
    #plt.ylabel('Probability')
    #plt.xlabel('Availabilities km/h')
    #plt.show()

    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    dt = 0.1
    fs = 1/dt
    fc = 2  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')

    random_agent_idx = np.random.randint(0, len(agent_dataset), size=1000)

    while True:
        for i, agent_dataset_idx  in enumerate(random_agent_idx):

            #agent_dataset_idx = 270
            data = agent_dataset[agent_dataset_idx]

            # Plot rasterizer
            fig, axs = plt.subplots(2, 3)
            #visualize_rgb_image(agent_dataset, index=agent_dataset_idx, title="py_satellite", ax=axs[0, 0])
            rs = cfg["raster_params"]["raster_size"]
            ec = cfg["raster_params"]["ego_center"]
            bias = np.array([rs[0] * ec[0], rs[1] * ec[1]])
            im = data["image"].transpose(1, 2, 0)
            im = agent_dataset.rasterizer.to_rgb(im)
            target_positions_pixels = data["target_positions"] + bias #transform_points(data["target_positions"] + data["centroid"][:2], data["agent_from_world"])
            draw_trajectory(im, target_positions_pixels, rgb_color=TARGET_POINTS_COLOR, radius=1, yaws=data["target_yaws"])
            # Plotting all the agents around the ego agent (note multiplying by 2 fro meters to pixel)
            history_all_agents = data["history_all_agents_positions"][:]
            for i_agent in range(history_all_agents.shape[0]):
                history_agent_x = 2*history_all_agents[i_agent, :, 0] + bias[0]
                history_agent_y = 2*history_all_agents[i_agent, :, 1] + bias[1]
                axs[0, 0].scatter(history_agent_x, history_agent_y, color="red", s=1, label='ground truth hist')
            axs[0, 0].imshow(im) # Why [::-1]

            # Plot position of first agent
            pos_hist = data["history_positions"] + data["centroid"][:2]
            pos_fut = data["target_positions"] + data["centroid"][:2]
            axs[0, 1].scatter(pos_hist[:, 0], pos_hist[:, 1], color="blue")
            axs[0, 1].scatter(pos_fut[:, 0], pos_fut[:, 1], color="green")
            axs[0, 1].axis('equal')

            # Calculate speed of car states

            # Velocity (note [::-1] in the position
            pos = np.vstack((data["history_positions"][::-1], data["target_positions"]))
            #pos = pos[~np.all(pos == 0, axis=1)] # Remove zeros
            ds = np.sqrt(np.sum(np.square(np.diff(pos, axis=0)), axis=1))
            velocity = 3.6 * ds/dt# km/h
            velocity_f = moving_average(velocity, 2) # moving average filter
            #velocity_LPf = signal.filtfilt(b, a, velocity, axis=0)
            axs[1, 0].plot(velocity, color="blue")
            axs[1, 0].plot(velocity_f, color="red")
            #axs[1, 0].plot(velocity_LPf, color="green")
            axs[1, 0].set_title('Speed at reference frame {:.2f}'.format(3.6*math.sqrt(data["velocity"][0]**2 + data["velocity"][1]**2)))

            # Yaw rate
            yaw_hist = data["history_yaws"][::-1] * 180/math.pi
            yaw_fut = data["target_yaws"]* 180/ math.pi
            axs[1, 2].plot(yaw_hist, color="blue")
            axs[1, 2].plot(yaw_fut, color="red")
            axs[1, 2].set_title('yaw [deg]')

            # Yaw rate
            yaw = np.vstack((data["history_yaws"][::-1], data["target_yaws"]))
            yaw = yaw[~np.all(yaw == 0, axis=1)] # Remove zeros
            dyaw = np.sqrt(np.square(np.diff(yaw, axis=0)))
            yaw_rate = dyaw/dt
            yaw_rate_f = moving_average(yaw_rate, 2)# moving average filter
            #yaw_rate_LPf = signal.filtfilt(b, a, yaw_rate, axis=0)
            axs[1, 1].plot(yaw_rate, color="blue")
            axs[1, 1].plot(yaw_rate_f, color="red")
            #axs[1, 1].plot(yaw_rate_LPf, color="green")
            axs[1, 1].set_title('yaw rate[rad/s]')

            # Acceleration
            dv = np.diff(velocity)
            acc = dv/dt# m/s2
            acc_f = moving_average(acc, 5) # moving average filter
            #acc_LPf = signal.filtfilt(b, a, acc, axis=0)
            axs[0, 2].plot(acc, color="blue")
            axs[0, 2].plot(acc_f, color="red")
            #axs[0, 2].plot(acc_LPf, color="green")
            axs[0, 2].set_title('accx [m/s]')

            # Lateral speed

            # Lateral acceleration
            if(keyboard.is_pressed('q')):
                break
            fig.show()
            plt.show()
        print('stop plotting')
        break

