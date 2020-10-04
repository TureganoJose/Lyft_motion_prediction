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
#import pandas as pd
from tqdm import tqdm
from typing import Dict
from collections import Counter
from prettytable import PrettyTable
import bisect
from L5_classes_experiments import CustomAgentDataset, CustomEgoDataset
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
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood

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

# models
from models import LyftMultiModel, forward, LyftMultiModel_carstates


# %% [code]
cfg = {
    'model_params': {
        'model_architecture': 'resnet101',
        'local_coordinates': True,
        'history_num_frames': 10,
        'lr': 1e-4,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'weight_path': "/kaggle/input/lyftpretrained-resnet101/lyft_resnet101_model.pth",
        'future_delta_time': 0.1,
        'train': True
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
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0
    },

    'sample_data_loader': {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr',
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0
    },
    "valid_data_loader": {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr',
         "batch_size": 1,
         "shuffle": True,
         "num_workers": 0},
    'test_data_loader': {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'checkpoint_every_n_steps': 5000,
        'max_num_steps': 10000
    },

    'test_params': {
        'image_coords': True

    }
}

if __name__ == '__main__':
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

    # ===== LOAD MODEL
    model = LyftMultiModel(cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.load_state_dict(torch.load("0918_predictor_full.pt", map_location=device)) #"lyft_multimode.pth"
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # ===== LOAD VALIDATION DATASET
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)
    valid_cfg = cfg["train_data_loader"]
    validate_zarr = ChunkedDataset(dm.require(valid_cfg["key"])).open()
    valid_dataset = AgentDataset(cfg, validate_zarr, rasterizer)
    #valid_sub_dataset = torch.utils.data.Subset(valid_dataset, range(100))
    valid_dataloader = DataLoader(valid_dataset, shuffle=valid_cfg["shuffle"], batch_size=valid_cfg["batch_size"],
                                 num_workers=valid_cfg["num_workers"])
    print("==================================VALIDATION DATA==================================")
    print(valid_dataset)

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []
    target_ls = []
    avails_ids = []
    progress_bar = tqdm(valid_dataloader)

    for data in progress_bar:
        inputs = data['image'].to(device)
        #car_states = data['car_states'].to(device)
        ims = data["image"].numpy().transpose(0, 2, 3, 1)#1, 2, 0)

        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)

        preds, confidences = model(inputs)#, car_states)
        future_coords_offsets_pd.append(preds.cpu().numpy().copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        pos = np.concatenate((data["history_positions"].numpy()[::-1], data["target_positions"].numpy()),axis=1)
        pos_hist = data["history_positions"].numpy()[::-1]
        pos_fut = data["target_positions"].numpy()
        #car_states = car_states.cpu().numpy()
        target_availabilities = target_availabilities.cpu().numpy()

        # convert agent coordinates into world offsets
        agents_preds = preds.cpu().numpy()
        preds_confidence = confidences.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        world_to_image = data["world_to_image"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []

        for agents_pred, conf_pred, agent_hist_gt, agent_fut_gt, world_from_agent, world_to_image, centroid, im,  target_availabilities in zip(agents_preds, preds_confidence, pos_hist, pos_fut, world_from_agents, world_to_image, centroids, ims, target_availabilities):
            im = valid_dataset.rasterizer.to_rgb(im)

            # Ground truth from new to old coordinates
            agent_hist_gt = transform_points(agent_hist_gt, world_from_agent) - centroid[:2]
            agent_fut_gt = transform_points(agent_fut_gt, world_from_agent) - centroid[:2]

            # Everything from old world to image coordinates
            agent_hist_gt_im = transform_points(agent_hist_gt + centroid[:2], world_to_image)
            agent_fut_gt_im = transform_points(agent_fut_gt + centroid[:2], world_to_image)
            agents_pred0_im = transform_points(agents_pred[0, :, :] + centroid[:2], world_to_image)
            agents_pred1_im = transform_points(agents_pred[1, :, :] + centroid[:2], world_to_image)
            agents_pred2_im = transform_points(agents_pred[2, :, :] + centroid[:2], world_to_image)

            # Calculate loss
            loss = neg_multi_log_likelihood(agent_fut_gt, agents_pred, conf_pred, target_availabilities[:, 0])

            # Plot
            if loss>50:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.imshow(im) #[::-1]
                ax1.scatter(agent_hist_gt_im[:, 0], agent_hist_gt_im[:, 1], color="blue", label='ground truth hist')
                ax1.scatter(agent_fut_gt_im[:, 0], agent_fut_gt_im[:, 1], color="green", label='ground truth fut')
                ax1.scatter(agents_pred0_im[:, 0], agents_pred0_im[:, 1], c='black', cmap='Reds', label='prediction 1 '+str(conf_pred[0]))
                ax1.scatter(agents_pred1_im[:, 0], agents_pred1_im[:, 1], c='grey', cmap='Reds', label='prediction 2 '+str(conf_pred[1]))
                ax1.scatter(agents_pred2_im[:, 0], agents_pred2_im[:, 1], c='red', cmap='Reds', label='prediction 3 '+str(conf_pred[2]))
                plt.legend(loc='upper left')

                plt.title('loss {:.2f}'.format(loss))# car speed{:.2f} yaw hist avg {:2f} acc avg {:.2f} size {:.2f}'.format(loss, car_states[0],car_states[1],car_states[2],car_states[3]))
                plt.show()
                print('')
                input("Press enter for the next plot")






        # timestamps.append(data["timestamp"].numpy().copy())
        # agent_ids.append(data["track_id"].numpy().copy())
        # target_ls.append(data["target_positions"].numpy().copy())
        # avails_ids.append(data["target_availabilities"].numpy().copy())

