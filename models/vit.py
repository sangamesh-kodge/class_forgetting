import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from collections import OrderedDict
from copy import deepcopy
from .model_utils import reshape_conv_input_activation, forward_cache_activations, forward_cache_projections, forward_cache_svd
import numpy as np


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )



class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        x, _ = self.self_attention(x, x, x, need_weights=False)
        # x = self.test_unrolled(x) # This verifies the correct unrolling of the above function. 
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
    def test_unrolled(self, x):
        # Ingore dropout as representation collected only in eval mode.
        bsz, seq_len, embed_dim  = x.shape
        q, k, v = F._in_projection_packed(x, x, x, self.self_attention.in_proj_weight, self.self_attention.in_proj_bias)
        q = q.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        k = k.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        v = v.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        attn_output = F.scaled_dot_product_attention(q,k,v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(seq_len*bsz, embed_dim)
        attn_output = F.linear(attn_output, self.self_attention.out_proj.weight, self.self_attention.out_proj.bias)
        attn_output = attn_output.view(bsz, seq_len, embed_dim)
        return attn_output
        

    def get_activations(self, input,  block_key, max_samples=10000):      
        act=OrderedDict()
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)        
        ### Unroll attention here! 
        bsz, seq_len, embed_dim  = x.shape

        act[f"{block_key}.self_attn.qkv"] = x.clone().detach().view(-1,  x.shape[-1])
        indexes = np.arange(act[f"{block_key}.self_attn.qkv"].shape[0])
        np.random.shuffle(indexes)
        act[f"{block_key}.self_attn.qkv"] = act[f"{block_key}.self_attn.qkv"][indexes[:max_samples]]
        q, k, v = F._in_projection_packed(x, x, x, self.self_attention.in_proj_weight, self.self_attention.in_proj_bias)
        q = q.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        k = k.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        v = v.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        attn_output = F.scaled_dot_product_attention(q,k,v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(seq_len*bsz, embed_dim)
        act[f"{block_key}.self_attn.out_proj"] = attn_output.clone().detach().view(-1,  attn_output.shape[-1])
        indexes = np.arange(act[f"{block_key}.self_attn.out_proj"].shape[0])
        np.random.shuffle(indexes)
        act[f"{block_key}.self_attn.out_proj"] = act[f"{block_key}.self_attn.out_proj"][indexes[:max_samples]]
        attn_output = F.linear(attn_output, self.self_attention.out_proj.weight, self.self_attention.out_proj.bias)
            
        
        x = attn_output.view(bsz, seq_len, embed_dim)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)        
        mlp_ind = 0
        for layer in self.mlp:            
            if isinstance(layer, nn.Linear):
                 layer_name = f"{block_key}.mlp.linear{mlp_ind}"
                 mlp_ind+=1
            else:
                layer_name = ""
            y_reshaped, layer_acts  = forward_cache_activations(y.view(-1,  y.shape[-1]), layer, layer_name, max_samples)
            act.update(layer_acts)  
            y = y_reshaped.view( (y.shape[0], y.shape[1], -1) )
            
        return act, x+y 
    
    def get_svd_directions(self, input,  block_key, max_samples=10000):      
        U=OrderedDict()
        S=OrderedDict()
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)        
        ### Unroll attention here! 
        bsz, seq_len, embed_dim  = x.shape


        activation =  x.clone().detach().view(-1,  x.shape[-1])
        indexes = np.arange(activation.shape[0])
        np.random.shuffle(indexes)
        activation = activation[indexes[:max_samples]].transpose(0,1)
        u,s,_ = torch.linalg.svd(activation, full_matrices=False)
        U[f"{block_key}.self_attn.qkv"] = u
        S[f"{block_key}.self_attn.qkv"] = s
        
        q, k, v = F._in_projection_packed(x, x, x, self.self_attention.in_proj_weight, self.self_attention.in_proj_bias)
        q = q.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        k = k.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        v = v.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        attn_output = F.scaled_dot_product_attention(q,k,v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(seq_len*bsz, embed_dim)

        activation =  attn_output.clone().detach().view(-1,  attn_output.shape[-1])
        indexes = np.arange(activation.shape[0])
        np.random.shuffle(indexes)
        activation = activation[indexes[:max_samples]].transpose(0,1)
        u,s,_ = torch.linalg.svd(activation, full_matrices=False)
        U[f"{block_key}.self_attn.out_proj"] = u
        S[f"{block_key}.self_attn.out_proj"] = s

        attn_output = F.linear(attn_output, self.self_attention.out_proj.weight, self.self_attention.out_proj.bias)
            
        
        x = attn_output.view(bsz, seq_len, embed_dim)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)        
        mlp_ind = 0
        for layer in self.mlp:            
            if isinstance(layer, nn.Linear):
                 layer_name = f"{block_key}.mlp.linear{mlp_ind}"
                 mlp_ind+=1
            else:
                layer_name = ""
            y_reshaped, layer_u, layer_s  = forward_cache_svd(y.view(-1,  y.shape[-1]), layer, layer_name,  max_samples)
            U.update(layer_u)  
            S.update(layer_s)  
            
            y = y_reshaped.view( (y.shape[0], y.shape[1], -1) )
            
        return U, S, x+y 
     

    def get_scaled_projections(self, input,  block_key, alpha, max_samples=10000):      
        proj = OrderedDict()
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)        
        ### Unroll attention here! 
        bsz, seq_len, embed_dim  = x.shape


        activation =  x.clone().detach().view(-1,  x.shape[-1])
        indexes = np.arange(activation.shape[0])
        np.random.shuffle(indexes)
        activation = activation[indexes[:max_samples]].transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        sval_total = (Sr**2).sum()
        sval_ratio = (Sr**2)/sval_total
        importance_r =  torch.diag( alpha *sval_ratio/((alpha-1)*sval_ratio+1) )
        mr = torch.mm( Ur, importance_r )
        proj[f"{block_key}.self_attn.qkv"] =  torch.mm( mr, Ur.transpose(0,1) )


        q, k, v = F._in_projection_packed(x, x, x, self.self_attention.in_proj_weight, self.self_attention.in_proj_bias)
        q = q.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        k = k.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        v = v.view(bsz, seq_len, self.num_heads, embed_dim//self.num_heads).permute(0,2,1,3)
        attn_output = F.scaled_dot_product_attention(q,k,v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(seq_len*bsz, embed_dim)

        activation =  attn_output.clone().detach().view(-1,  attn_output.shape[-1])
        indexes = np.arange(activation.shape[0])
        np.random.shuffle(indexes)
        activation = activation[indexes[:max_samples]].transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        sval_total = (Sr**2).sum()
        sval_ratio = (Sr**2)/sval_total
        importance_r =  torch.diag( alpha *sval_ratio/((alpha-1)*sval_ratio+1) )
        mr = torch.mm( Ur, importance_r )
        proj[f"{block_key}.self_attn.out_proj"] =  torch.mm( mr, Ur.transpose(0,1) )

        attn_output = F.linear(attn_output, self.self_attention.out_proj.weight, self.self_attention.out_proj.bias)
            
        
        x = attn_output.view(bsz, seq_len, embed_dim)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)        
        mlp_ind = 0
        for layer in self.mlp:            
            if isinstance(layer, nn.Linear):
                 layer_name = f"{block_key}.mlp.linear{mlp_ind}"
                 mlp_ind+=1
            else:
                layer_name = ""
            y_reshaped, layer_proj  = forward_cache_projections(y.view(-1,  y.shape[-1]), layer, layer_name, alpha, max_samples)
            proj.update(layer_proj)  
            y = y_reshaped.view( (y.shape[0], y.shape[1], -1) )
            
        return proj, x+y 

    def project_weights(self,  projection_mat_dict, block_key):
        w_q, w_k, w_v = self.self_attention.in_proj_weight.chunk(3)
        w_q.data = torch.mm(w_q.data, projection_mat_dict[f"{block_key}.self_attn.qkv"].transpose(0,1) ).view_as(w_q.data)
        w_k.data = torch.mm(w_k.data, projection_mat_dict[f"{block_key}.self_attn.qkv"].transpose(0,1) ).view_as(w_k.data)
        w_v.data = torch.mm(w_v.data, projection_mat_dict[f"{block_key}.self_attn.qkv"].transpose(0,1) ).view_as(w_v.data)
        self.self_attention.in_proj_weight.data = torch.cat((w_q.data, w_k.data, w_v.data ), 0) 
        
        self.self_attention.out_proj.weight.data = torch.mm(self.self_attention.out_proj.weight.data, projection_mat_dict[f"{block_key}.self_attn.out_proj"].transpose(0,1) ).view_as(w_v.data)
        mlp_ind = 0
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict[f"{block_key}.mlp.linear{mlp_ind}"].transpose(0,1)).view_as(layer.weight.data)
                mlp_ind+=1
        return


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))    
    
    def get_activations(self, input):      
        act=OrderedDict()
        encoderblock_ind=0
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        for layer in self.layers:
            block_name = f"encoderblock{encoderblock_ind}"
            encoderblock_ind+=1
            layer_acts, input = layer.get_activations(input, block_name)
            act.update(layer_acts)    
        return act, self.ln(input)
    
        
    def get_svd_directions(self, input, max_samples=10000):      
        U=OrderedDict()
        S=OrderedDict()
        encoderblock_ind=0
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        for layer in self.layers:
            block_name = f"encoderblock{encoderblock_ind}"
            encoderblock_ind+=1
            layer_u, layer_s, input = layer.get_svd_directions(input, block_name, max_samples)
            U.update(layer_u)    
            S.update(layer_s)    
        return U, S, self.ln(input)

    def get_scaled_projections(self, input, alpha, max_samples=10000):      
        proj=OrderedDict()
        encoderblock_ind=0
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        for layer in self.layers:
            block_name = f"encoderblock{encoderblock_ind}"
            encoderblock_ind+=1
            layer_proj, input = layer.get_scaled_projections(input, block_name, alpha, max_samples)
            proj.update(layer_proj)    
        return proj, self.ln(input)    

    
    def project_weights(self, projection_mat_dict):
        encoderblock_ind=0
        for layer in self.layers:
            block_name = f"encoderblock{encoderblock_ind}"
            encoderblock_ind+=1
            layer.project_weights(projection_mat_dict, block_name)
        return


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        
        x = self.heads(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def get_activations(self, x, max_samples=10000):      
        conv_proj_ind=0
        heads_ind=0
        ### x = self._process_input(x)
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # x = self.conv_proj(x)
        if isinstance(self.conv_proj, nn.Conv2d):
            x, act  = forward_cache_activations(x, self.conv_proj, f"conv_proj.conv{conv_proj_ind}", max_samples)  
            conv_proj_ind+=1
        else:
            act=OrderedDict()
            for layer in self.conv_proj:
                if isinstance(layer, nn.Conv2d):
                    layer_name = f"conv_proj.conv{conv_proj_ind}"
                    conv_proj_ind+=1
                else:
                    layer_name = ""
                x, layer_acts  = forward_cache_activations(x, layer, f"conv_proj.conv{conv_proj_ind}", max_samples)  
                act.update(layer_acts)    
                    
        x = x.reshape(n, self.hidden_dim, n_h * n_w)        

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        encoder_activation, x = self.encoder.get_activations(x)
        act.update(encoder_activation)  

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if isinstance(self.heads, nn.Linear):    
            x, layer_acts  = forward_cache_activations(x, self.heads, f"heads.linear{heads_ind}", max_samples)  
            act.update(layer_acts)   
            heads_ind+=1
        else:
            for layer in self.heads:
                if isinstance(layer, nn.Linear):
                    layer_name = f"heads.linear{heads_ind}"
                    heads_ind+=1
                else:
                    layer_name = ""
                x, layer_acts  = forward_cache_activations(x, layer, layer_name, max_samples)  
                act.update(layer_acts)  
        self.max_head_linear_layers = heads_ind
        return act
    

    def get_svd_directions(self, x,  max_samples=10000):      
        conv_proj_ind=0
        heads_ind=0
        ### x = self._process_input(x)
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # x = self.conv_proj(x)
        if isinstance(self.conv_proj, nn.Conv2d):
            x, U, S  = forward_cache_svd(x, self.conv_proj, f"conv_proj.conv{conv_proj_ind}",  max_samples)  
            conv_proj_ind+=1
        else:
            U=OrderedDict()
            S=OrderedDict()
            
            for layer in self.conv_proj:
                if isinstance(layer, nn.Conv2d):
                    layer_name = f"conv_proj.conv{conv_proj_ind}"
                    conv_proj_ind+=1
                else:
                    layer_name = ""
                x, layer_u, layer_s  = forward_cache_svd(x, layer, f"conv_proj.conv{conv_proj_ind}",  max_samples)  
                U.update(layer_u)    
                S.update(layer_s)    
                    
        x = x.reshape(n, self.hidden_dim, n_h * n_w)        

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        encoder_u, encoder_s, x = self.encoder.get_svd_directions(x, max_samples)
        U.update(encoder_u)  
        S.update(encoder_s)  
        

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if isinstance(self.heads, nn.Linear):    
            x, layer_u, layer_s  = forward_cache_svd(x, self.heads, f"heads.linear{heads_ind}",  max_samples)  
            U.update(layer_u)  
            S.update(layer_s)   
            heads_ind+=1
        else:
            for layer in self.heads:
                if isinstance(layer, nn.Linear):
                    layer_name = f"heads.linear{heads_ind}"
                    heads_ind+=1
                else:
                    layer_name = ""
                x, layer_u, layer_s  = forward_cache_svd(x, layer, layer_name,  max_samples)  
                U.update(layer_u)  
                S.update(layer_s)   
        self.max_head_linear_layers = heads_ind
        return U, S
    
    

    def get_scaled_projections(self, x, alpha, max_samples=10000):      
        conv_proj_ind=0
        heads_ind=0
        ### x = self._process_input(x)
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # x = self.conv_proj(x)
        if isinstance(self.conv_proj, nn.Conv2d):
            x, proj  = forward_cache_projections(x, self.conv_proj, f"conv_proj.conv{conv_proj_ind}", alpha, max_samples)  
            conv_proj_ind+=1
        else:
            proj=OrderedDict()
            for layer in self.conv_proj:
                if isinstance(layer, nn.Conv2d):
                    layer_name = f"conv_proj.conv{conv_proj_ind}"
                    conv_proj_ind+=1
                else:
                    layer_name = ""
                x, layer_proj  = forward_cache_projections(x, layer, f"conv_proj.conv{conv_proj_ind}", alpha, max_samples)  
                proj.update(layer_proj)    
                    
        x = x.reshape(n, self.hidden_dim, n_h * n_w)        

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        encoder_proj, x = self.encoder.get_scaled_projections(x, alpha, max_samples)
        proj.update(encoder_proj)  

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if isinstance(self.heads, nn.Linear):    
            x, layer_proj  = forward_cache_projections(x, self.heads, f"heads.linear{heads_ind}", alpha, max_samples)  
            proj.update(layer_proj)   
            heads_ind+=1
        else:
            for layer in self.heads:
                if isinstance(layer, nn.Linear):
                    layer_name = f"heads.linear{heads_ind}"
                    heads_ind+=1
                else:
                    layer_name = ""
                x, layer_proj  = forward_cache_projections(x, layer, layer_name, alpha, max_samples)  
                proj.update(layer_proj)  
        self.max_head_linear_layers = heads_ind
        return proj
    
    def project_weights(self, projection_mat_dict, proj_classifier=False):
        conv_proj_ind=0
        heads_ind=0
        if isinstance(self.conv_proj, nn.Conv2d):
            self.conv_proj.weight.data = torch.mm(self.conv_proj.weight.data.flatten(1), projection_mat_dict[f"conv_proj.conv{conv_proj_ind}"].transpose(0,1)).view_as(self.conv_proj.weight.data)
            conv_proj_ind+=1
        else:
            for layer in self.conv_proj:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict[f"conv_proj.conv{conv_proj_ind}"].transpose(0,1)).view_as(layer.weight.data)
                    
                    conv_proj_ind+=1
    
                    
        self.encoder.project_weights(projection_mat_dict)

        if isinstance(self.heads, nn.Linear):
            self.heads.weight.data  = torch.mm(self.heads.weight.data , projection_mat_dict[f"heads.linear{heads_ind}"].transpose(0,1) )
            heads_ind+=1
        else:
            for layer in self.heads:
                if isinstance(layer, nn.Linear):
                    layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict[f"heads.linear{heads_ind}"].transpose(0,1)).view_as(layer.weight.data)
                    heads_ind+=1
        return 
    

def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_ops": 55.484,
            "_file_size": 331.398,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88224232,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                }
            },
            "_ops": 4.409,
            "_file_size": 336.604,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=242),
        meta={
            **_COMMON_META,
            "num_params": 304326632,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        transforms=partial(
            ImageClassification,
            crop_size=512,
            resize_size=512,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 305174504,
            "min_size": (512, 512),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                }
            },
            "_ops": 361.986,
            "_file_size": 1164.258,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 304326632,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                }
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_32-c7638314.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 306535400,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                }
            },
            "_ops": 15.378,
            "_file_size": 1169.449,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        transforms=partial(
            ImageClassification,
            crop_size=518,
            resize_size=518,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 633470440,
            "min_size": (518, 518),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                }
            },
            "_ops": 1016.717,
            "_file_size": 2416.643,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 632045800,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                }
            },
            "_ops": 167.295,
            "_file_size": 2411.209,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_SWAG_E2E_V1


# @register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


# @register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_32_Weights.IMAGENET1K_V1))
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    weights = ViT_B_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


# @register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_16_Weights.IMAGENET1K_V1))
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    weights = ViT_L_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


# @register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_32_Weights.IMAGENET1K_V1))
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    weights = ViT_L_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


# @register_model()
@handle_legacy_interface(weights=("pretrained", None))
def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    weights = ViT_H_14_Weights.verify(weights)

    return _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state
