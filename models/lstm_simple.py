import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18,resnet50,resnet101
from tqdm import tqdm
from typing import Dict
from torch import functional as F


class EncoderLSTM_LyftModel(nn.Module):

    def __init__(self, cfg):
        super(EncoderLSTM_LyftModel, self).__init__()

        self.input_sz = 2
        self.hidden_sz = 128
        self.num_layer = 1
        self.sequence_length = 11

        self.Encoder_lstm = nn.LSTM(self.input_sz, self.hidden_sz, self.num_layer, batch_first=True)

    def forward(self, inputs):
        output, hidden_state = self.Encoder_lstm(inputs)

        return output, hidden_state


class DecoderLSTM_LyftModel(nn.Module):
    def __init__(self, cfg):
        super(DecoderLSTM_LyftModel, self).__init__()

        self.input_sz = 128  # (2000 from fcn_en_output reshape to 50*40)
        self.hidden_sz = 128
        self.hidden_sz_en = 128
        self.num_layer = 1
        self.sequence_len_de = 1

        self.interlayer = 256

        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        self.encoderLSTM = EncoderLSTM_LyftModel(cfg)

        self.Decoder_lstm = nn.LSTM(self.input_sz, self.hidden_sz, self.num_layer, batch_first=True)

        self.fcn_en_state_dec_state = nn.Sequential(
            nn.Linear(in_features=self.hidden_sz_en, out_features=self.interlayer),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.interlayer, out_features=num_targets))

    def forward(self, inputs):
        _, hidden_state = self.encoderLSTM(inputs)

        inout_to_dec = torch.ones(inputs.shape[0], self.sequence_len_de, self.input_sz).to(device)

        # for i in range(cfg["model_params"]["future_num_frames"]+1): # this can be used to feed output from previous LSTM to anther one which is stacked.
        inout_to_dec, hidden_state = self.Decoder_lstm(inout_to_dec, (hidden_state[0], hidden_state[1]))

        fc_out = self.fcn_en_state_dec_state(inout_to_dec.squeeze(dim=0))

        return fc_out.reshape(inputs.shape[0], cfg["model_params"]["future_num_frames"], -1)