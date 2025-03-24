import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os

from mamba_ssm import Mamba2, Mamba


class CompressMambaBlock(nn.Module):
    def __init__(self, device, d_model, head_dim, output_dim, expand, dropout_rate=0.2):
        super(CompressMambaBlock, self).__init__()
        self.device = device
        self.d_model = d_model
        self.head_dim = head_dim
        self.expand = expand
        self.dropout_rate = dropout_rate

        self.mamba = Mamba2(d_model, expand=expand, headdim=head_dim, device=device)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        if device is not None:
            self.to(device)

    def forward(self, x):
        x = self.mamba(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_norm2(x)
        return x


class CompressMambaBackbone(nn.Module):
    def __init__(self, device, d_model, head_dim, expand, mamba_blockcnt=8, dropout_rate=0.2):
        super(CompressMambaBackbone, self).__init__()
        self.device = device
        self.d_model = d_model
        self.head_dim = head_dim
        self.expand = expand
        mamba_blocks = [
            CompressMambaBlock(device, d_model, head_dim, d_model, expand, dropout_rate)
            for _ in range(mamba_blockcnt)
        ]
        self.mamba_backbone = nn.Sequential(*mamba_blocks)
        
    def forward(self, x):
        x = self.mamba_backbone(x)
        return x


class MLP(nn.Module):
    def __init__(self, device, input_dim, output_dim, hidden_dim=128, dropout_rate=0.2, num_layers=2):
        super(MLP, self).__init__()
        self.fc_layers = [nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)]
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_layers = [nn.Dropout(dropout_rate) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        if device is not None:
            self.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.fc_layers[i](x))
            x = self.dropout_layers[i](x)
            x = self.layer_norm(x)
        x = self.output_layer(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.rot_matrix_params = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.orthogonal_(self.rot_matrix_params)

    def forward(self, x, global_seg_position=0):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len, batch_size = x.size(1), x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(-1, self.d_model)
        position_encoding = torch.cos(positions * 0.01)
        position_encoding = position_encoding.unsqueeze(0).expand(batch_size, -1, -1).to(self.rot_matrix_params.dtype)

        if torch.is_tensor(global_seg_position):
            global_position = global_seg_position.unsqueeze(1).unsqueeze(2)
            global_position = global_position.expand(-1, seq_len, self.d_model)
            global_position_encoding = torch.cos(global_position * 0.01).to(self.rot_matrix_params.dtype)
            position_encoding = position_encoding + global_position_encoding

        x = x + position_encoding 
        x = torch.matmul(x, self.rot_matrix_params) 

        return x


class CompressEmbedding(nn.Module):
    def __init__(self, device, vocab_size, hidden_dim=128):
        super(CompressEmbedding, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = RotaryPositionalEmbedding(hidden_dim)
        self.global_position_embedding = RotaryPositionalEmbedding(hidden_dim)
        if device is not None:
            self.to(device)

    def forward(self, x, global_seg_position=0):
        x = self.embedding(x)
        x = self.position_embedding(x)
        x = self.global_position_embedding(x, global_seg_position)
        return x


class CompressMamba(nn.Module):
    def __init__(self, output_dim, vocab_size, device=None, mamba_blocks=24, expand=4, d_model=768, head_dim=768, output_heads=5):
        super(CompressMamba, self).__init__()
        self.embedding = CompressEmbedding(device, vocab_size, d_model)
        # Mamba Backbone
        self.mamba_blocks = CompressMambaBackbone(device, d_model, head_dim, expand, mamba_blocks)
        self.layer_norm = nn.LayerNorm(d_model)
        if device is not None:
            self.layer_norm.to(device)

        # MLP
        self.mlps = [MLP(device, d_model, output_dim, hidden_dim=d_model) for _ in range(output_heads)]
        self.mlps = nn.ModuleList(self.mlps)

    def forward(self, x, global_seg_position=0):
        x = self.embedding(x, global_seg_position)
        x = self.mamba_blocks(x)
        x = self.layer_norm(x)
        x = x[:, -1, :].view(-1, x.size(-1))
        x = [mlp(x) for mlp in self.mlps]
        return x
