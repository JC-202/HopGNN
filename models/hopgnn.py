from torch import nn
import sys
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class Interaction_GCN(nn.Module):
    def __init__(self, hidden_channels):
        super(Interaction_GCN, self).__init__()
        self.fc = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, inputs):
        x = inputs.mean(dim=1, keepdim=True)
        return self.fc(x)

class Interaction_SAGE(nn.Module):
    def __init__(self, hidden_channels):
        super(Interaction_SAGE, self).__init__()
        self.fc_l = nn.Linear(hidden_channels, hidden_channels)
        self.fc_r = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, inputs):
        neighbor = inputs.mean(dim=1, keepdim=True)
        neighbor = self.fc_r(neighbor)
        x = self.fc_l(inputs)
        x = (x + neighbor)
        return x

class Interaction_Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class HopGNN(torch.nn.Module):
    def __init__(self, g, in_channels, hidden_channels, out_channels, num_hop=6, dropout=0.5, activation='relu',
                 feature_inter='attention', inter_layer=2, feature_fusion='attention', norm_type='ln'):
        super().__init__()
        self.num_hop = num_hop
        self.feature_inter_type = feature_inter
        self.feature_fusion = feature_fusion
        self.dropout = nn.Dropout(dropout)
        self.pre = False
        self.g = g
        self.norm_type = norm_type
        self.build_activation(activation)

        #encoder
        self.fc = nn.Linear(in_channels, hidden_channels)

        #hop_embedding
        self.hop_embedding = nn.Parameter(torch.randn(1, num_hop, hidden_channels))
        #interaction
        self.build_feature_inter_layer(feature_inter, hidden_channels, inter_layer)

        #fusion
        if self.feature_fusion == 'attention':
            self.atten_self = nn.Linear(hidden_channels, 1)
            self.atten_neighbor = nn.Linear(hidden_channels, 1)

        #prediction
        self.classifier = nn.Linear(hidden_channels, out_channels)

        #norm
        self.build_norm_layer(hidden_channels, inter_layer * 2 + 2)
        print('HopGNN hidden:', hidden_channels, 'interaction:', feature_inter, 'hop:',num_hop, 'layers:',inter_layer)

    def build_activation(self, activation):
        if activation == 'tanh':
            self.activate = F.tanh
        elif activation == 'sigmoid':
            self.activate = F.sigmoid
        elif activation == 'gelu':
            self.activate = F.gelu
        else:
            self.activate = F.relu

    def preprocess(self, adj, x):
        h0 = []
        for i in range(self.num_hop):
            h0.append(x)
            x = adj @ x
        self.h0 = torch.stack(h0, dim=1)
        self.pre = True
        return self.h0

    def build_feature_inter_layer(self, feature_inter, hidden_channels, inter_layer):
        self.interaction_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        if feature_inter == 'mlp':
            for i in range(inter_layer):
                mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
                self.interaction_layers.append(mlp)
        elif feature_inter == 'gcn':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_GCN(hidden_channels))
        elif feature_inter == 'sage':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_SAGE(hidden_channels))
        elif feature_inter == 'attention':
            for i in range(inter_layer):
                self.interaction_layers.append(
                    Interaction_Attention(hidden_channels, heads=4, dropout=0.1))
        else:
            self.interaction_layers.append(torch.nn.Identity())


    def build_norm_layer(self, hidden_channels, layers):
        self.norm_layers = nn.ModuleList()
        for i in range(layers):
            if self.norm_type == 'bn':
                self.norm_layers.append(nn.BatchNorm1d(self.num_hop))
            elif self.norm_type == 'ln':
                self.norm_layers.append(nn.LayerNorm(hidden_channels))
            else:
                self.norm_layers.append(nn.Identity())

    def norm(self, h, layer_index):
        h = self.norm_layers[layer_index](h)
        return h

    # N * hop * d => N * hop * d
    def embedding(self, h):
        h = self.dropout(h)
        h = self.fc(h)
        h = h + self.hop_embedding
        h = self.norm(h, 0)
        return h

    # N * hop * d =>  N * hop * d
    def interaction(self, h):
        inter_layers = len(self.interaction_layers)
        for i in range(inter_layers):
            h_prev = h
            h = self.dropout(h)
            h = self.interaction_layers[i](h)
            h = self.activate(h)
            h = h + h_prev
            h = self.norm(h, i + 1)
        return h

    # N * hop * d =>  N * hop * d (concat) or N * d (mean/max/attention)
    def fusion(self, h):
        h = self.dropout(h)
        if self.feature_fusion == 'max':
            h = h.max(dim=1).values
        elif self.feature_fusion == 'attention':
            h_self, h_neighbor = h[:, 0, :], h[:, 1:, :]
            h_self_atten = self.atten_self(h_self).view(-1, 1)
            h_neighbor_atten = self.atten_neighbor(h_neighbor).squeeze()
            h_atten = torch.softmax(F.leaky_relu(h_self_atten+h_neighbor_atten), dim=1)
            h_neighbor = torch.einsum('nhd, nh -> nd', h_neighbor, h_atten).squeeze()
            h = h_self + h_neighbor
        else: #mean
            h = h.mean(dim=1)
        h = self.norm(h, -1)
        return h

    def build_hop(self, inputs):
        if len(inputs.shape) == 3:
            h = inputs
        else:
            if self.pre == False:
                self.h0 = self.preprocess(self.g, inputs)
            h = self.h0
        return h

    def forward(self, inputs):
        # step-1 the first preprocess of hop-information for accerelate training
        h = self.build_hop(inputs)
        # step-2 hop-embedding
        h = self.embedding(h)
        # step-3 hop-interaction
        h = self.interaction(h)
        # step-4 hop-fusion
        h = self.fusion(h)
        # step-5 prediction
        h = self.classifier(h)
        return h

    #HopGNN+ with self-supervised learning (barlow twins)
    def forward_plus(self, inputs):
        h = self.build_hop(inputs)
        #generate two views along batch dimension
        aug_h = torch.cat((h, h), dim=0)

        #augmentation inside with dropout
        h = self.embedding(aug_h)
        h = self.interaction(h)
        h = self.fusion(h)
        y = self.classifier(h)

        #split two views of embedding and outputs
        size = h.size(0)
        y1, y2 = y[:size // 2, ...], y[size // 2:, ...]
        view1, view2 = h[:size // 2, ...], h[size // 2:, ...]
        return (y1, y2), (view1, view2)
