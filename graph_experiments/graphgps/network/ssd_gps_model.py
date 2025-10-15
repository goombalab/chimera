import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import product

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import from_networkx, to_networkx

import networkx as nx
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from graphgps.layer.ssd_gps_layer import SSDGPSLayer


class Normalizer(nn.Module):
    def __init__(self, norm_style, out_dim):
        super(Normalizer, self).__init__()
        if norm_style == "batch":
            # Use BatchNorm1d for batch normalization
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_style == "layer":
            # Use PyG LayerNorm for layer normalization
            self.norm = pygnn.norm.LayerNorm(out_dim)
        elif norm_style == "none":
            # Identity function for no normalization
            self.norm = nn.Identity()
        self.norm_style = norm_style

    def forward(self, batch):
        # For LayerNorm, pass batch information; otherwise, ignore batch
        if self.norm_style == "layer":
            batch.x = self.norm(batch.x, batch.batch)
        else:
            batch.x = self.norm(batch.x)
        return batch


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if "PNA" in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg
                    )
                )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("SSDGPSModel")
class SSDGPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )
        layers = []
        self.add_conv_layers = cfg.gt.add_conv_layers
        if self.add_conv_layers:
            for _ in range(cfg.gt.conv_layers):
                layers.append(
                    GCNConvLayer(
                        dim_in,
                        dim_in,
                        dropout=cfg.gt.dropout,
                        residual=True,
                    )
                )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split("+")
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")

        for _ in range(cfg.gt.layers):
            layers.append(
                SSDGPSLayer(
                    dim_h=cfg.gt.dim_hidden,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_model_type,
                    num_heads=cfg.gt.n_heads,
                    act=cfg.gnn.act,
                    pna_degrees=cfg.gt.pna_degrees,
                    equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    norm_style=cfg.gt.norm_style,
                    local_norm_style=cfg.gt.local_norm_style,
                    norm_position=cfg.gt.norm_position,
                    is_block_composed=cfg.gt.is_block_composed,
                    is_ffn_block=cfg.gt.is_ffn_block,
                    num_dags=cfg.gt.num_dags,
                    layer_args=cfg.gssd.layer_args,
                    d_inner=cfg.gt.d_inner,
                    ffn_d_dinner=cfg.gt.ffn_d_dinner,
                    d_state=cfg.gt.d_state,
                    use_dag_decomposition=cfg.gt.use_dag_decomposition,
                )
            )

        self.extra_normalizer = cfg.gt.extra_normalizer
        if self.extra_normalizer:
            self.normalizer = Normalizer(cfg.gt.norm_style, cfg.gt.dim_hidden)
            layers.append(self.normalizer)

        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
        self.if_dag_decompose = cfg.gt.use_dag_decomposition
        self.num_dags = cfg.gt.num_dags

    def forward(self, batch):
        # Add a "batch_edge" attribute that corresponds to the batch_id
        # of a given edge.
        source_batch = batch.batch[batch.edge_index[0]]  # Graph IDs of source nodes
        target_batch = batch.batch[batch.edge_index[1]]  # Graph IDs of target nodes
        assert torch.equal(
            source_batch, target_batch
        ), "Edges must connect nodes from the same graph in a batched graph."
        batch.batch_edge = source_batch

        dag_masks = None
        if self.if_dag_decompose:
            # As a hack, dag_masks are present in the last num_dags of edge_attr
            dag_masks = (
                batch.edge_attr[:, -self.num_dags :].to(torch.bool).transpose(0, 1)
            )
            assert len(dag_masks) == self.num_dags
            batch.dag_masks = dag_masks
            batch.edge_attr = batch.edge_attr[:, : -self.num_dags]

        for module in self.children():
            batch = module(batch)
        return batch  #
