from loss_functions import pytorch_neg_multi_log_likelihood_batch

from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler

from classes.custom_classes import CustomAgentDataset, CustomEgoDataset
import numpy as np
from tqdm import tqdm

import os

PATH_TO_DATA = "../input/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = PATH_TO_DATA

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
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace


import torch
from torch import nn, optim

from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from torchvision.models import mobilenet_v2
from torch import nn
from torch import Tensor
from typing import Dict
from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Subset
from models.stamina_net import TemporalEncoderLSTM, SpatialEncoderLSTM, raster_encoder
from models.stamina_net import STAMINA_net, attention_mechanism, MultiHeadAttention, decoder
from models.stamina_res_net import STAMINA_res_net

import pickle



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
        'shuffle': False,
        'num_workers': 0
    },

    'sample_data_loader': {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr',
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    },
    "valid_data_loader": {
        'key': 'C:\Workspaces\L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr',
         "batch_size": 32,
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

class IndexSampler(Sampler):
    def __init__(self, index, is_shuffle=False):
        #self.dataset = dataset
        self.index = index
        self.is_shuffle = is_shuffle
        self.len = len(index)

    def __iter__(self):
        index = self.index.copy()
        if self.is_shuffle:
            random.shuffle(index)
        return iter(index)

    def __len__(self):
        return self.len


def train(device, model, train_dataloader, valid_dataloader, opt=None, criterion=None):
    """Function for training the model"""
    print("Training...")
    progress = tqdm(range(cfg["train_params"]["max_num_steps"]))
    train_iter = iter(train_dataloader)
    val_iter = []  # iter(valid_dataloader)

    for i in progress:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            data = next(train_iter)

        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        model.train()
        torch.set_grad_enabled(True)

        # Forward pass
        outputs, confidences = stamina(data)
        loss = criterion(targets, outputs, confidences, target_availabilities.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        if VALIDATION:
            with torch.no_grad():
                try:
                    val_data = next(val_iter)
                except StopIteration:
                    val_iter = iter(valid_dataloader)
                    val_data = next(val_iter)
                val_outputs, val_confidences = stamina(data)
                val_loss = criterion(targets, val_outputs, val_confidences, target_availabilities)
            desc = f"Loss: {round(loss.item(), 4)} Validation Loss: {round(val_loss.item(), 4)}"
        else:
            desc = f"Loss: {round(loss.item(), 4)}"

        # Save checkpoint
        if (i + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0:
            torch.save({'epoch': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'stamina_{i+0}.pth')
        progress.set_description(desc)

    return model


VALIDATION = False  # A hyperparameter you could use to toggle for validating the model
PRETRAINED = True
last_used = 56500

# %% [code]
if __name__ == '__main__':

    index_scenes = []
    if not PRETRAINED:
        index_scenes = np.arange(3000) #531365
        np.random.shuffle(index_scenes)
        with open('index_scenes.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(index_scenes, f)
    else:
        with open('index_scenes.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            index_scenes = pickle.load(f)

    # ===== LOAD TRAINING DATASET
    dm = LocalDataManager(None)
    train_cfg = cfg["sample_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = CustomAgentDataset(cfg, train_zarr, rasterizer)
    # train_sub_dataset = torch.utils.data.Subset(train_dataset, range(100))
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], sampler=IndexSampler(index_scenes[last_used:]),
                                  batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    print("==================================TRAIN DATA==================================")
    print(train_dataset)

    if VALIDATION:
        # ===== LOAD VALIDATION DATASET
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

    # ==== TRAIN LOOP
    device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    stamina = STAMINA_res_net(cfg,
                          n_agents=100,
                          n_car_states=7,
                          n_frames=11,
                          batch_size=1,
                          device=device,
                          spatial_en_hid_size=128,
                          temp_en_hid_size=128,
                          temp_att_embed_dim=128,
                          temp_n_heads=16,
                          spatial_att_embed_dim=128,
                          spatial_n_heads=16,
                          map_att_embed_dim=128,
                          map_n_heads=16,
                          map_k_dim=1,
                          map_v_dim=1)
    stamina.to(device)

    optimizer = optim.Adam(stamina.parameters(), lr=cfg["model_params"]["lr"])

    if PRETRAINED:
        WEIGHT_FILE = 'stamina_res_56499.pth'
        checkpoint = torch.load(WEIGHT_FILE, map_location=device)
        epoch = checkpoint['epoch']
        stamina.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    train_accuracy, val_accuracy, losses, val_losses, model = train(device=device,
                                                                    model=stamina,
                                                                    train_dataloader=train_dataloader,
                                                                    valid_dataloader=valid_dataloader,
                                                                    opt=optimizer,
                                                                    criterion=pytorch_neg_multi_log_likelihood_batch)

    # ===== SAVING MODEL
    print("Saving the model...")
    torch.save(model.state_dict(), "stamina.pth")
    print('model saved')

