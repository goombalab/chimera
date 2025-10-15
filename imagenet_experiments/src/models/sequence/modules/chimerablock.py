import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from src.models.nn import Activation, DropoutNd
from mamba_ssm.ops.triton.layernorm import RMSNorm
from .chimera import Chimera


class ChimeraBlock(nn.Module):
    def __init__(
        self,
        # model size configs
        d_model,
        qk_dim=64,
        expand_factor="2.0",
        headdim=64,
        # graph mamba flags
        unified_view=False,
        include_headnodes="1111", # order: top_left, top_right, bottom_left, bottom_right
        debug_use_get_A_dpr=False,
        debug_store_mm=False,
        share_BC=False,
        share_BC_for_two_graphs=True,
        share_dt_for_two_graphs=True,
        share_BC_for_two_graphs_mode="line", # "diagonal"
        use_fast_inverse=True,
        dt_min_max_factor=1.0,
        dt_self_min_max_factor=1.0,
        normalization_mode="dt_original", # "dt_self", "sqrt"
        add_fc_layers=False,
        ff_expand_factor=4,
        norm_sqrt_mul_factor=1.0, # < 1
        # other configs
        d_conv=2,
        conv_init=None,
        activation='swish',
        bias=False,
        dropout=0.0,
        tie_dropout=False,
        device=None,
        dtype=None,
        image_height=14, #Need to support dynamic image height
        image_width=14, #Need to support dynamic image width
        **unused_args,
    ):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.d_model = d_model
        self.qk_dim = qk_dim  
        self.expand_factor = expand_factor
        self.headdim= headdim

        # graph mamba flags
        self.unified_view = unified_view
        self.include_headnodes = include_headnodes
        self.debug_use_get_A_dpr = debug_use_get_A_dpr
        self.debug_store_mm = debug_store_mm
        self.share_BC = share_BC
        self.share_BC_for_two_graphs = share_BC_for_two_graphs
        self.share_dt_for_two_graphs = share_dt_for_two_graphs
        self.share_BC_for_two_graphs_mode = share_BC_for_two_graphs_mode
        self.add_fc_layers = add_fc_layers
        self.ff_expand_factor = ff_expand_factor
        self.use_fast_inverse = use_fast_inverse
        self.dt_min_max_factor = dt_min_max_factor
        self.dt_self_min_max_factor = dt_self_min_max_factor
        self.normalization_mode = normalization_mode
        self.norm_sqrt_mul_factor = norm_sqrt_mul_factor
        
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.activation = activation
        self.bias = bias
        self.image_height = image_height
        self.image_width = image_width

        assert activation in ['swish', 'silu']
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        
        if "/" in expand_factor:
            parts = expand_factor.split("/")
            numerator = float(parts[0])
            denominator = float(parts[1])
            self.expand_factor = numerator / denominator
        else:
            self.expand_factor = float(expand_factor)
        self.d_inner = round(self.expand_factor * self.d_model)


        assert self.d_inner % self.headdim == 0
        self.num_heads = self.d_inner // self.headdim

        # Build the sequence mixer
        self.mixer = Chimera(
            d_model=self.d_model,
            qk_dim=self.qk_dim,
            expand_factor=self.expand_factor,
            headdim=self.headdim,
            unified_view=self.unified_view,
            include_headnodes=self.include_headnodes,
            debug_use_get_A_dpr=self.debug_use_get_A_dpr,
            debug_store_mm=self.debug_store_mm,
            share_BC=self.share_BC,
            share_BC_for_two_graphs=self.share_BC_for_two_graphs,
            share_dt_for_two_graphs=self.share_dt_for_two_graphs,
            share_BC_for_two_graphs_mode=self.share_BC_for_two_graphs_mode,
            use_fast_inverse=self.use_fast_inverse,
            dt_min_max_factor=self.dt_min_max_factor,
            dt_self_min_max_factor=self.dt_self_min_max_factor,
            normalization_mode=self.normalization_mode,
            norm_sqrt_mul_factor=self.norm_sqrt_mul_factor,
            image_height=self.image_height,
            image_width=self.image_width)
        self.act = Activation(self.activation)

        self.num_b_or_c = 1 if self.share_BC or self.unified_view else sum(map(int, self.include_headnodes))
        if self.share_BC_for_two_graphs:
           assert self.share_BC == False
           assert  sum(map(int, self.include_headnodes)) == 4
           self.num_b_or_c = self.num_b_or_c - 2
            
        self.num_dt = sum(map(int, self.include_headnodes))
        if self.share_dt_for_two_graphs:
            assert self.share_BC_for_two_graphs == True
            self.num_dt = self.num_dt - 2
        self.num_dt_self = 1 if self.unified_view else sum(map(int, self.include_headnodes))
        self.is_add_dt_self = self.normalization_mode == "dt_self"

        d_in_proj = (
            self.d_inner + self.d_inner
            + (self.qk_dim*2*self.num_b_or_c)
            + self.num_heads*2*self.num_dt
            + self.num_heads*self.num_dt_self*self.is_add_dt_self
        )
        conv_dim = self.d_inner + self.qk_dim*2*self.num_b_or_c

        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,       
            out_channels=conv_dim,      
            bias=True,              
            kernel_size=(self.d_conv * 2 - 1, self.d_conv * 2 - 1), 
            groups=conv_dim,
            padding=(self.d_conv - 1, self.d_conv - 1),
            **factory_kwargs
        )
        # # Mask for directed convolution
        # self.directed_conv_mask = torch.zeros((self.d_conv * 2 - 1, self.d_conv * 2 - 1))
        # if self.include_headnodes[0] == '1':
        #     self.directed_conv_mask[0: self.d_conv, 0: self.d_conv] = torch.flip(torch.tril(
        #         torch.ones(self.d_conv, self.d_conv)), dims=[1])
        # if self.include_headnodes[1] == '1':
        #     self.directed_conv_mask[0: self.d_conv, self.d_conv - 1:] = torch.tril(
        #         torch.ones(self.d_conv, self.d_conv))
        # if self.include_headnodes[2] == '1':
        #     self.directed_conv_mask[self.d_conv - 1:, 0: self.d_conv] = torch.triu(
        #         torch.ones(self.d_conv, self.d_conv))
        # if self.include_headnodes[3] == '1':
        #     self.directed_conv_mask[self.d_conv - 1:, self.d_conv - 1:] = torch.flip(torch.triu(
        #         torch.ones(self.d_conv, self.d_conv)), dims=[1])

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.norm = RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        if self.add_fc_layers:
            self.fc_blocks = nn.Sequential(
                nn.Linear(self.d_model, self.ff_expand_factor*self.d_model),
                nn.SiLU(),
                nn.Linear(self.ff_expand_factor*self.d_model, d_model)
            )
            self.fc_dropout = nn.Dropout(dropout)
            self.fc_layer_norm = nn.LayerNorm(d_model)

    def forward(self, u, state=None, bias=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        L = u.size(-2)
        device = u.device

        zxBCdt = self.in_proj(u)
        if self.normalization_mode == "dt_self":
            z, xBC, dt, dt_self = torch.split(
                zxBCdt, [
                    self.d_inner,
                    self.d_inner + (self.qk_dim*2*self.num_b_or_c),
                    self.num_heads*2*self.num_dt,
                    self.num_heads*self.num_dt_self,
                ], dim=-1
            )
        else:
            z, xBC, dt = torch.split(
                zxBCdt, [
                    self.d_inner,
                    self.d_inner + (self.qk_dim*2*self.num_b_or_c),
                    self.num_heads*2*self.num_dt,
                ], dim=-1
            )
            dt_self = None

        # shape of |xBC| is "b (i j) d"
        xBC = rearrange(
            xBC, 
            'b (i j) d -> b d i j', 
            i = self.image_height,
            j = self.image_width,
        )
        # self.directed_conv_mask = self.directed_conv_mask.to(device)
        # self.conv2d.weight.data *= self.directed_conv_mask
        xBC = self.act(self.conv2d(xBC))
        xBC = rearrange(
            xBC, 
            'b d i j-> b (i j) d', 
            i = self.image_height,
            j = self.image_width,
        )
        x, B, C = torch.split(
            xBC, [self.d_inner, self.num_b_or_c*self.qk_dim, self.num_b_or_c*self.qk_dim], dim=-1)

        y = self.mixer(x, dt, [B,C], dt_self)
        y = self.norm(y, z) # y = rmsnorm(y) * F.silu(z)
        y = self.dropout(y)

        #y could be in fp32 because of the SSMs
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.out_proj.weight.dtype)
        y = self.out_proj(y)
        y = y[:, :L, :]

        if self.add_fc_layers:
            residual = y
            y = self.fc_blocks(y)
            y = self.fc_dropout(y)
            y = self.fc_layer_norm(y + residual)

        return y, None

    @property
    def d_output(self):
        return self.d_model
