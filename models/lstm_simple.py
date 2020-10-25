import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18,resnet50,resnet101
from tqdm import tqdm
from typing import Dict
from torch import functional as F
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0, n_frame_history=11, batch_size=32, device='cpu'):
    super(EncoderLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.n_frame_history = n_frame_history
    self._device = device
    self._batch_size = batch_size

    self.embedding = nn.Linear(input_size, hidden_size)
    self.lstm_encoder = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=False)

  def forward(self, inputs, hidden):
    embedded_input = Variable(torch.zeros(self.n_frame_history, self._batch_size, self.hidden_size)).to(self._device)
    output = Variable(torch.zeros(self.n_frame_history, self._batch_size, self.hidden_size)).to(self._device)
    # Embed input agents vector
    for iframe in range(self.n_frame_history):
        embedded_input[iframe, :, :] = self.embedding(inputs[:, :, iframe])
        # Pass the embedded word vectors into LSTM and return all outputs
    output, hidden = self.lstm_encoder(embedded_input, hidden)
    return output, hidden

  def init_hidden(self, batch_size=1):
    return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self._device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self._device))



class raster_encoder(nn.Module):
    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34
        # And it is 2048 for the other resnets
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        #self.num_preds = num_targets * num_modes
        #self.num_modes = num_modes
#
        #self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        #x = self.logit(x)

        ## pred (batch_size)x(modes)x(time)x(2D coords)
        ## confidences (batch_size)x(modes)
        #bs, _ = x.shape
        #pred, confidences = torch.split(x, self.num_preds, dim=1)
        #pred = pred.view(bs, self.num_modes, self.future_len, 2)
        #assert confidences.shape == (bs, self.num_modes)
        #confidences = torch.softmax(confidences, dim=1)
        return x














class EncoderLSTM_LyftModel(nn.Module):

    def __init__(self, cfg):
        super(EncoderLSTM_LyftModel, self).__init__()

        self.input_sz = 2
        self.hidden_sz = 128
        self.num_layer = 1
        self.sequence_length = 11


        self.embedding_input = nn.Linear(self.input_sz, self.hidden_sz)
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