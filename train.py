from loss_functions import pytorch_neg_multi_log_likelihood_batch

from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler

from classes.custom_classes import CustomAgentDataset, CustomEgoDataset
import numpy as np
from tqdm import tqdm

import os

PATH_TO_DATA = "/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles"
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
from models.stamina_no_ras_net import STAMINA_no_ras_net
from validation_function import validation
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
        'satellite_map_key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/aerial_map/aerial_map.png',
        'semantic_map_key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb',
        'dataset_meta_key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/meta.json',
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces': False
    },

    'train_data_loader': {
        'key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 2
    },

    'sample_data_loader': {
        'key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 2
    },
    "valid_data_loader": {
        'key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr',
         "batch_size": 64,
         "shuffle": False,
         "num_workers": 2
    },
    'test_data_loader': {
        'key': '/media/jose/OS/Workspaces/L5_competition/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 2
    },

    'train_params': {
        'checkpoint_every_n_steps': 100,
        'max_num_steps': 500
    },

    'valid_params': {
        'checkpoint_every_n_steps': 100,
        'max_num_steps': 500
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


def train(device, model, train_dataloader, valid_dataloader, opt=None, criterion=None, validation_function=None):
    """Function for training the model"""
    print("Training...")
    progress = tqdm(range(cfg["train_params"]["max_num_steps"]))
    train_iter = iter(train_dataloader)
    val_iter = []  # iter(valid_dataloader)
    total_loss =[]

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
        outputs, confidences = model(data)
        loss = criterion(targets, outputs, confidences, target_availabilities.squeeze(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Validation
        if VALIDATION and (i + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0:
            val_loss = validation_function('/home/jose/Repos/Lyft_motion_prediction/gt.csv',
                                            model,
                                            valid_dataloader)
            desc = f"Loss: {round(loss.item(), 4)} Validation Loss: {round(val_loss.item(), 4)}"
            losses.append(loss.item())
            vals.append(val_loss.item())
        else:
            desc = f"Loss: {round(loss.item(), 4)} Total loss: {round(loss.item(), 4)}"
            losses.append(loss.item())

        # Save checkpoint
        if (i + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0:
            torch.save({'i_sample': i + 1,
                        'losses': losses,
                        'vals': vals,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'stamina_{i+0}.pth')
        progress.set_description(desc)

    return model


VALIDATION = True  # A hyperparameter you could use to toggle for validating the model
PRETRAINED = False

# %% [code]
if __name__ == '__main__':

    # ==== LOAD PARAMETERS
    train_cfg = cfg["train_data_loader"]
    eval_cfg = cfg["valid_data_loader"]

    # ==== TRAIN LOOP
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    stamina = STAMINA_no_ras_net(cfg,
                          n_agents=100,
                          n_car_states=7,
                          n_frames=11,
                          batch_size=train_cfg["batch_size"],
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
    losses = []
    vals = []
    i_sample = 1

    if PRETRAINED:
        WEIGHT_FILE = 'stamina_res_56499.pth'
        checkpoint = torch.load(WEIGHT_FILE, map_location=device)
        i_sample = checkpoint['iter']
        stamina.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        vals = checkpoint['vals']

        # ===== LOAD TRAINING DATASET
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = CustomAgentDataset(cfg, train_zarr, rasterizer)

    index_scenes = []
    if not PRETRAINED:
        index_scenes = np.arange(round(len(train_dataset)/train_cfg["batch_size"]))  # 531365
        np.random.shuffle(index_scenes)
        with open('index_scenes.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(index_scenes, f)
    else:
        with open('index_scenes.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            index_scenes = pickle.load(f)

    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"],
                                  sampler=IndexSampler(index_scenes[i_sample:]), drop_last=True,
                                  batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    print("==================================TRAIN DATA==================================")
    print(train_dataset)

    if VALIDATION:
        ## ===== LOAD VALIDATION DATASET

        # Creating data indices for training and validation splits:
        eval_zarr_path ='/home/jose/Repos/Lyft_motion_prediction/validate.zarr'
        eval_mask_path='/home/jose/Repos/Lyft_motion_prediction/mask.npz'
        eval_zarr = ChunkedDataset(eval_zarr_path).open()
        eval_mask = np.load(eval_mask_path)["arr_0"]
        # ===== INIT DATASET AND LOAD MASK
        eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
        dataset_size = len(eval_dataset)
        indices = list(range(dataset_size))
        split = cfg["valid_params"]["max_num_steps"]
        random_seed = 42 + 5
        shuffle_dataset=False
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        val_indices = indices[:split]

        valid_sampler = SubsetRandomSampler(val_indices)
        val_dataloader = DataLoader(eval_dataset, sampler=valid_sampler, drop_last=True,
                                     batch_size=cfg["valid_data_loader"]["batch_size"],
                                     num_workers=cfg["valid_data_loader"]["num_workers"])

        print(val_dataloader)
        print("==================================VALIDATION DATA==================================")

    else:
        valid_dataloader = []

    train_accuracy, val_accuracy, losses, val_losses, model = train(device=device,
                                                                    model=stamina,
                                                                    train_dataloader=train_dataloader,
                                                                    valid_dataloader=val_dataloader,
                                                                    opt=optimizer,
                                                                    criterion=pytorch_neg_multi_log_likelihood_batch,
                                                                    validation_function=validation)

    # ===== SAVING MODEL
    print("Saving the model...")
    torch.save(model.state_dict(), "stamina.pth")
    print('model saved')

