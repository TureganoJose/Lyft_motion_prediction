import keyboard

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gc
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

# loss functions
from loss_functions import pytorch_neg_multi_log_likelihood_batch
# %% [code]
cfg = {
    'model_params': {
        'model_architecture': 'resnet18',
        'history_num_frames': 10,
        'lr': 1e-3,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
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
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    },

    'sample_data_loader': {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    },
    "valid_data_loader": {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr',
         "batch_size": 1,
         "shuffle": False,
         "num_workers": 0},
    'test_data_loader': {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'checkpoint_every_n_steps': 1000,
        'max_num_steps': 5000
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
    dm = LocalDataManager(None)
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    # train_sub_dataset = torch.utils.data.Subset(train_dataset, range(100))
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                  num_workers=train_cfg["num_workers"])
    print("==================================TRAIN DATA==================================")
    print(train_dataset)

    VALIDATION = False

    # ===== LOAD VALIDATION DATASET
    if VALIDATION:
        valid_cfg = cfg["valid_data_loader"]
        validate_zarr = ChunkedDataset(dm.require(valid_cfg["key"])).open()
        valid_dataset = CustomAgentDataset(cfg, validate_zarr, rasterizer)
        # valid_sub_dataset = torch.utils.data.Subset(valid_dataset, range(100))
        valid_dataloader = DataLoader(valid_dataset, shuffle=valid_cfg["shuffle"], batch_size=valid_cfg["batch_size"],
                                      num_workers=valid_cfg["num_workers"])
        print("==================================VALIDATION DATA==================================")
        print(valid_dataset)
    else:
        valid_dataloader = []
    # %% [code]

    # ==== TRAIN LOOP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LyftMultiModel(cfg)

    model.load_state_dict(torch.load("resnet18_baseline_1999.pth", map_location=device))
    # weight_path = cfg["model_params"]["weight_path"]
    # model.load_state_dict(torch.load(weight_path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    # optimizer = AdamP(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-2)#optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    print(f'device {device}')


    # %% [code]
    def train(model, train_dataloader, valid_dataloader, opt=None, criterion=None, lrate=1e-4):
        """Function for training the model"""
        print("Building Model...")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = LyftMultiModel(cfg)
        #
        # model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
        criterion = pytorch_neg_multi_log_likelihood_batch

        print("Training...")
        losses = []
        train_accuracy = []
        train_accuracy_ave = []
        losses_mean = []
        val_accuracy = []
        val_accuracy_ave = []
        val_losses = []
        val_losses_mean = []
        progress = tqdm(range(cfg["train_params"]["max_num_steps"]))  #
        train_iter = iter(train_dataloader)
        val_iter = iter(valid_dataloader)

        for i in progress:
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                data = next(train_iter)

            model.train()
            torch.set_grad_enabled(True)

            # Forward pass
            inputs = data["image"].to(device)
            target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
            targets = data["target_positions"].to(device)
            outputs, confidences = model(inputs)
            loss = criterion(targets, outputs, confidences, target_availabilities[:,: , 0])

            #acc, loss, _ = forward(data, model, device, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #train_accuracy.append(acc.item())
            #train_accuracy_ave.append(np.mean(train_accuracy))

            #losses.append(loss.item())
            #losses_mean.append(np.mean(losses))
            losses_mean=[]

            # Validation
            if VALIDATION:
                with torch.no_grad():
                    try:
                        val_data = next(val_iter)
                    except StopIteration:
                        val_iter = iter(valid_dataloader)
                        val_data = next(val_iter)

                    val_acc, val_loss, _ = forward(val_data, model, device, criterion)
                    val_losses.append(val_loss.item())
                    val_losses_mean.append(np.mean(val_losses))
                    val_accuracy.append(val_acc.item())
                    val_accuracy_ave.append(np.mean(val_accuracy))
                desc = f"Loss: {round(loss.item(), 4)} Validation Loss: {round(val_loss.item(), 4)}"
            else:
                val_losses_mean = []
                val_accuracy_ave = []
                train_accuracy_ave = []
                desc = f"Loss: {round(loss.item(), 4)}"


            # Saving checkpoint
            if (i + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0:
                torch.save({'epoch': i + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           f'resnet18_baseline_{i}.pth')

                # if len(losses)>0 and loss < min(losses):
            #    print(f"Loss improved from {min(losses)} to {loss}")
            gc.collect()
            progress.set_description(desc)

        return train_accuracy_ave, val_accuracy_ave, losses_mean, val_losses_mean, model


    train_accuracy, val_accuracy, losses, val_losses, model = train(model, train_dataloader, valid_dataloader, criterion=pytorch_neg_multi_log_likelihood_batch)

    print("losses mean {:.2f}", losses)
    # ===== SAVING MODEL
    print("Saving the model...")
    torch.save(model.state_dict(), "resnet18_baseline.pth")
    print('model saved')