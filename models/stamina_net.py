import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18,resnet50,resnet101
from tqdm import tqdm
from typing import Dict
from torch import functional as F
from torch.autograd import Variable
import math

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_len):
    """Perform softmax by filtering out some elements."""
    # X: 3-D tensor, valid_len: 1-D or 2-D tensor
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, repeats=shape[1],
                                                dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_len, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `query`: (`batch_size`, #queries, `d`)
    # `key`: (`batch_size`, #kv_pairs, `d`)
    # `value`: (`batch_size`, #kv_pairs, `dim_v`)
    # `valid_len`: either (`batch_size`, ) or (`batch_size`, xx)
    def forward(self, query, key, value, valid_len=None):
        d = query.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of key
        scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return torch.bmm(attention_weights, value)

def transpose_qkv(X, num_heads):
    # Input `X` shape: (`batch_size`, `seq_len`, `num_hiddens`).
    # Output `X` shape:
    # (`batch_size`, `seq_len`, `num_heads`, `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # `X` shape:
    # (`batch_size`, `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # `output` shape:
    # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output


#@save
def transpose_output(X, num_heads):
    # A reversed version of `transpose_qkv`
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, query, key, value, valid_len):
        # For self-attention, `query`, `key`, and `value` shape:
        # (`batch_size`, `seq_len`, `dim`), where `seq_len` is the length of
        # input sequence. `valid_len` shape is either (`batch_size`, ) or
        # (`batch_size`, `seq_len`).

        # Project and transpose `query`, `key`, and `value` from
        # (`batch_size`, `seq_len`, `num_hiddens`) to
        # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_len is not None:
            if valid_len.ndim == 1:
              valid_len = valid_len.repeat(self.num_heads)
            else:
              valid_len = valid_len.repeat(self.num_heads, 1)

        # For self-attention, `output` shape:
        # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
        output = self.attention(query, key, value, valid_len)

        # `output_concat` shape: (`batch_size`, `seq_len`, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



class TemporalEncoderLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0, n_frame_history=11, batch_size=32, device='cpu'):
    super(TemporalEncoderLSTM, self).__init__()
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

class SpatialEncoderLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0, n_car_states=7, n_agents=100, n_frame_history=11, batch_size=32, device='cpu'):
    super(SpatialEncoderLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.n_car_states = n_car_states
    self.n_agents = n_agents
    self.n_frame_history = n_frame_history
    self._device = device
    self._batch_size = batch_size

    self.embedding = nn.Linear(input_size, hidden_size)
    self.lstm_encoder = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=False)

  def forward(self, inputs, hidden):
    embedded_input = Variable(torch.zeros(self.n_agents, self._batch_size, self.hidden_size)).to(self._device)
    output = Variable(torch.zeros(self.n_agents, self._batch_size, self.hidden_size)).to(self._device)
    # Embed input agents vector
    for iagent in range(self.n_agents):
        embedded_input[iagent, :, :] = self.embedding(torch.reshape(inputs[:, self.n_car_states*(iagent):self.n_car_states*(iagent)+7, :], (self._batch_size, self.n_car_states*self.n_frame_history)))
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

        num_hist_frames = (cfg["model_params"]["history_num_frames"] + 1)  #including current frame
        num_history_channels = (num_hist_frames) * 2
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

        return x


class attention_mechanism(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, k_dim=None, v_dim=None):
        super(attention_mechanism, self).__init__()
        self.attention_mechanism = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, kdim=k_dim, vdim=v_dim)

    def forward(self, Q, K, V):
        '''query: (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        key: (S, N, E)(S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
        value: (S, N, E)(S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.'''

        output = self.attention_mechanism(Q, K, V)
        return output

class decoder(nn.Module):
    def __init__(self,  cfg: Dict,  input_size, hidden_size, n_layers=1, drop_prob=0, n_frame_history=11, batch_size=32,
                 device='cpu',num_modes=3):
        super(decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_frame_history = n_frame_history
        self._device = device
        self._batch_size = batch_size

        self.lstm_decoder = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=False)

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(n_frame_history*hidden_size, out_features=self.num_preds + num_modes)


    def forward(self, inputs, hidden):
        decoder_lstm_output = Variable(torch.zeros(self.n_frame_history, self._batch_size, self.hidden_size)).to(self._device)
        decoder_lstm_output, hidden = self.lstm_decoder(inputs, hidden)
        decoder_lstm_output = decoder_lstm_output.permute(1, 0, 2)
        decoder_lstm_output = torch.flatten(decoder_lstm_output, 1)
        x = self.logit(decoder_lstm_output)

        ## pred (batch_size)x(modes)x(time)x(2D coords)
        ## confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self._device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self._device))



class STAMINA_net(nn.Module):
    def __init__(self, cfg: Dict, n_agents=100, n_car_states=7, n_frames=11, batch_size=32, device='cpu',
                 spatial_en_hid_size=128, temp_en_hid_size=128,
                 temp_att_embed_dim=128, temp_n_heads=16, spatial_att_embed_dim=128, spatial_n_heads=16,
                 map_att_embed_dim=128, map_n_heads=16, map_k_dim=1, map_v_dim=1):
        ''' STAMinA Spacial-Temmporal Attention with Map indexing of near Agents '''
        """ This is my first net, STAMINA, Spatial-Temporal Attention with Maps Indexing Near Agents.
        The name is a stretch, I know, it's pretty much random and based on my personal intuition of what it might work,
        it's got 3 main branches: spacial, temporal and maps. First two branches have 2 encoders, one for all the agents
        and one for the reference vehicle (called ego agent here). The third map uses the map features in conjunction
         with the encoded ego agent states. Each of the branches have a multi-head attention module """

        super(STAMINA_net, self).__init__()
        # General parameters
        self.n_agents = n_agents
        self.n_car_states = n_car_states
        self.n_frames = n_frames
        self.device = device
        self.batch_size = batch_size

        # Spatial encoder parameters
        self.spatial_en_hid_size = spatial_en_hid_size
        # Temporal encoder parameters
        self.temp_en_hid_size = temp_en_hid_size

        self.total_concat_size = 3 * spatial_en_hid_size + 3 * temp_en_hid_size
        # ==== DEFINING MODEL
        # Temporal Encoders
        self.agents_encoder = TemporalEncoderLSTM(n_agents * n_car_states, temp_en_hid_size, n_frame_history=n_frames, batch_size=self.batch_size, device=self.device)
        self.ego_agent_encoder = TemporalEncoderLSTM(n_car_states, temp_en_hid_size, n_frame_history=n_frames,
                                                     batch_size=batch_size, device=device)

        # Temporal attention mechanisms
        self.temporal_attention = attention_mechanism(embed_dim=temp_att_embed_dim, num_heads=temp_n_heads)
        # temporal_attention = MultiHeadAttention(key_size=128, query_size=128, value_size=128, num_hiddens=128,
        #             num_heads=16, dropout=0.0, bias=False, valid_len=None).to(device)

        # Spatial encoder
        self.spatial_agents_encoder = SpatialEncoderLSTM(n_frames * n_car_states, spatial_en_hid_size, n_car_states=n_car_states,
                                                         n_agents=n_agents, n_frame_history=n_frames, batch_size=batch_size,
                                                         device=device)
        self.spatial_ego_agent_encoder = SpatialEncoderLSTM(n_frames * n_car_states, spatial_en_hid_size, n_car_states=n_car_states,
                                                            n_agents=1, n_frame_history=n_frames, batch_size=batch_size,
                                                            device=device)

        # Spatial Attention mechanism
        self.Spatial_attention = attention_mechanism(embed_dim=spatial_att_embed_dim, num_heads=spatial_n_heads)

        # Map Encoder
        self.image_encoder = raster_encoder(cfg)

        # Map Attention mechanism
        self.map_attention = attention_mechanism(embed_dim=map_att_embed_dim, num_heads=map_n_heads, k_dim=map_k_dim, v_dim=map_v_dim)

        # Decoder
        self.final_layer = decoder(cfg, input_size=self.total_concat_size, hidden_size=128, n_layers=1, drop_prob=0,
                                   n_frame_history=n_frames,
                                   batch_size=batch_size, device=device, num_modes=3)



    def forward(self, input_data):

        # Input data
        input_data_agents = input_data["agents_state_vector"].to(self.device)
        input_data_ego_agent = input_data["ego_agent_state_vector"].to(self.device)
        input_image = input_data["image"].to(self.device)

        # Temporal encoding
        h_agents = self.agents_encoder.init_hidden(self.batch_size)
        h_ego_agent = self.ego_agent_encoder.init_hidden(self.batch_size)
        encoder_agents_outputs, h_agents = self.agents_encoder(input_data_agents.float(), h_agents)  # to(torch.int64)
        encoder_ego_agent_outputs, h_ego_agent = self.ego_agent_encoder(input_data_ego_agent.float(),
                                                                   h_ego_agent)  # to(torch.int64)

        attn_output = self.temporal_attention(encoder_ego_agent_outputs, encoder_agents_outputs, encoder_agents_outputs)

        # Spacial encoding
        h_spatial_agents = self.spatial_agents_encoder.init_hidden(self.batch_size)
        h_spatial_ego_agent = self.spatial_ego_agent_encoder.init_hidden(self.batch_size)
        Spatial_encoder_agents_outputs, h_spatial_agents = self.spatial_agents_encoder(input_data_agents.float(),
                                                                                       h_spatial_agents)  # to(torch.int64)
        Spatial_encoder_ego_agent_outputs, h_spatial_ego_agent = self.spatial_ego_agent_encoder(input_data_ego_agent.float(),
                                                                                                h_spatial_ego_agent)  # to(torch.int64)

        Spatial_attn_output = self.Spatial_attention(Spatial_encoder_ego_agent_outputs, Spatial_encoder_agents_outputs,
                                                     Spatial_encoder_agents_outputs)

        # Encoding map
        image_features = self.image_encoder(input_image)
        image_features = image_features.unsqueeze(0)
        image_features = image_features.permute(2, 1, 0)

        map_attn_output = self.map_attention(Spatial_encoder_ego_agent_outputs, image_features, image_features)

        # Concat all three: spatial, temporal and map
        concat_temporal_tensor = torch.cat((attn_output[0], encoder_ego_agent_outputs), dim=2)
        concat_spatial_tensor = torch.cat((Spatial_attn_output[0], Spatial_encoder_ego_agent_outputs), dim=2)
        concat_map_tensor = torch.cat((map_attn_output[0], Spatial_encoder_ego_agent_outputs), dim=2)

        # Temporal concat
        final_encoded_tensor = torch.zeros((self.n_frames, self.batch_size, self.total_concat_size))
        for iframe in range(self.n_frames):
            final_encoded_tensor[iframe, :, :] = torch.cat((concat_temporal_tensor[iframe, :, :].unsqueeze(0),
                                                            concat_spatial_tensor, concat_map_tensor), dim=2)

        h_final = self.final_layer.init_hidden(self.batch_size)
        predictions, probabilities = self.final_layer(final_encoded_tensor.to(self.device), h_final)

        return predictions, probabilities
