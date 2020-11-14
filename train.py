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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from models.stamina_net import DecoderLSTM_LyftModel, TemporalEncoderLSTM, SpatialEncoderLSTM, raster_encoder, attention_mechanism, MultiHeadAttention, decoder



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


def train(device, model, train_dataset, train_dataloader, valid_dataloader, opt=None, criterion=None, lrate=1e-4):
    """Function for training the model"""
    print("Training...")
    losses = []
    progress = tqdm(range(cfg["train_params"]["max_num_steps"]))
    train_iter = iter(train_dataloader)
    val_iter = []  # iter(valid_dataloader)

    for i in progress:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            data = next(train_iter)

        model.train()
        torch.set_grad_enabled(True)


        # Forward pass
        history_positions = data['history_positions'].to(device)
        all_history_positions = data['history_all_agents_positions'].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets, confidences = data["target_positions"].to(device)
        lengths = data["num_agents"].to(device)

        inputs = torch.nn.utils.rnn.pack_padded_sequence(history_positions, lengths, batch_first=True)
        outputs = model(history_positions)
        loss = criterion(targets, outputs, confidences, target_availabilities)
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


                # val_loss, _ = forward(val_data, model, device, criterion)
            desc = f"Loss: {round(loss.item(), 4)} Validation Loss: {round(val_loss.item(), 4)}"
        else:
            desc = f"Loss: {round(loss.item(), 4)}"

        # Save checkpoint
        if (i + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0:
            torch.save({'epoch': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'MNv2_carstates_{i}.pth')
        progress.set_description(desc)

    return model


VALIDATION = False  # A hyperparameter you could use to toggle for validating the model
PRETRAINED = False


# %% [code]
if __name__ == '__main__':
    # ===== LOAD TRAINING DATASET
    dm = LocalDataManager(None)
    train_cfg = cfg["sample_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = CustomAgentDataset(cfg, train_zarr, rasterizer)
    # train_sub_dataset = torch.utils.data.Subset(train_dataset, range(100))
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                  num_workers=train_cfg["num_workers"])
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hidden_size = 128
    n_car_states = 7
    n_agents = 100
    n_frames = 11
    batch_size = 32


















    # ==== DEFINING MODEL
    # Encoders
    agents_encoder = TemporalEncoderLSTM(n_agents*n_car_states, hidden_size, n_frame_history=n_frames, batch_size=32, device=device).to(device)
    ego_agent_encoder = TemporalEncoderLSTM(n_car_states, hidden_size, n_frame_history=n_frames, batch_size=32, device=device).to(device)


    print(agents_encoder)
    data = next(iter(train_dataloader))
    batch_size = data["agents_state_vector"].shape[0]

    h_agents = agents_encoder.init_hidden(batch_size)
    h_ego_agent = ego_agent_encoder.init_hidden(batch_size)

    input_data_agents = data["agents_state_vector"].to(device)
    input_data_ego_agent = data["ego_agent_state_vector"].to(device)
    input_image = data["image"].to(device)

    encoder_agents_outputs, h_agents = agents_encoder(input_data_agents.float(), h_agents) #to(torch.int64)
    encoder_ego_agent_outputs, h_ego_agent = ego_agent_encoder(input_data_ego_agent.float() , h_ego_agent) #to(torch.int64)

    # Temporal attention mechanisms
    temporal_attention = attention_mechanism(embed_dim=128, num_heads=16).to(device)
    #temporal_attention = MultiHeadAttention(key_size=128, query_size=128, value_size=128, num_hiddens=128,
    #             num_heads=16, dropout=0.0, bias=False, valid_len=None).to(device)
    attn_output = temporal_attention(encoder_ego_agent_outputs, encoder_agents_outputs, encoder_agents_outputs)

    # Spatial encoder
    spatial_agents_encoder = SpatialEncoderLSTM(n_frames*n_car_states, hidden_size, n_car_states=n_car_states, n_agents=n_agents, n_frame_history=n_frames, batch_size=batch_size, device=device).to(device)
    spatial_ego_agent_encoder = SpatialEncoderLSTM(n_frames*n_car_states, hidden_size, n_car_states=n_car_states, n_agents=1, n_frame_history=n_frames, batch_size=batch_size, device=device).to(device)

    h_spatial_agents = spatial_agents_encoder.init_hidden(batch_size)
    h_spatial_ego_agent = spatial_ego_agent_encoder.init_hidden(batch_size)

    Spatial_encoder_agents_outputs, h_Spatial_agents = spatial_agents_encoder(input_data_agents.float(), h_spatial_agents) #to(torch.int64)
    Spatial_encoder_ego_agent_outputs, h_Spatial_ego_agent = spatial_ego_agent_encoder(input_data_ego_agent.float() , h_spatial_ego_agent) #to(torch.int64)

    # Spatial Attention mechanism
    Spatial_attention = attention_mechanism(embed_dim=128, num_heads=16).to(device)
    Spatial_attn_output = Spatial_attention(Spatial_encoder_ego_agent_outputs, Spatial_encoder_agents_outputs, Spatial_encoder_agents_outputs)

    # Map Encoder
    image_encoder = raster_encoder(cfg).to(device)
    image_features = image_encoder(input_image)
    image_features = image_features.unsqueeze(0)
    image_features = image_features.permute(2, 1, 0)

    # Map Attention mechanism
    map_attention = attention_mechanism(embed_dim=128, num_heads=16, k_dim=1, v_dim=1).to(device)
    map_attn_output = map_attention(Spatial_encoder_ego_agent_outputs, image_features, image_features)

    # Concat all three: spatial, temporal and map
    concat_temporal_tensor = torch.cat((attn_output[0], encoder_ego_agent_outputs), dim=2)
    concat_spatial_tensor = torch.cat((Spatial_attn_output[0], Spatial_encoder_ego_agent_outputs), dim=2)
    concat_map_tensor = torch.cat((map_attn_output[0], Spatial_encoder_ego_agent_outputs), dim=2)

    # Temporal concat
    total_concat_size = concat_temporal_tensor.shape[2] + concat_spatial_tensor.shape[2] + concat_map_tensor.shape[2]
    final_encoded_tensor = torch.zeros((n_frames, batch_size, total_concat_size))
    for iframe in range(n_frames):
        final_encoded_tensor[iframe, :, :] = torch.cat((concat_temporal_tensor[iframe, :, :].unsqueeze(0),
                                                        concat_spatial_tensor, concat_map_tensor), dim=2)

    # Decoder
    final_layer = decoder( cfg,  input_size = total_concat_size, hidden_size= 128, n_layers=1, drop_prob=0, n_frame_history=n_frames,
                          batch_size=batch_size,device='cpu',num_modes=3)
    h_final = final_layer.init_hidden(batch_size)
    predictions, probabilities = final_layer(final_encoded_tensor, h_final)

    optimizer = optim.Adam(agents_encoder.parameters(), lr=cfg["model_params"]["lr"])
    agents_encoder.to(device)

    if PRETRAINED:
        WEIGHT_FILE = 'MNv2_carstates_4999.pth'
        checkpoint = torch.load(WEIGHT_FILE, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(model)

    criterion = pytorch_neg_multi_log_likelihood_batch












    model = train(device, model, train_dataset, train_dataloader, valid_dataloader, optimizer, criterion,
                  lrate=cfg['model_params']['lr'])

    # ===== SAVING MODEL
    print("Saving the model...")
    torch.save(model.state_dict(), "MNv2_carstates.pth")
    print('model saved')



    dm = LocalDataManager()
    dataset_path = dm.require(cfg["sample_data_loader"]["key"])
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

