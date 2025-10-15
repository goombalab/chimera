import warnings

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer, Normalizer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.graph_ssd_dag import DAGSSDLayer
from graphgps.layer.graph_ssd_for_gps_layer import GraphSSDLayer


class SSDGPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer."""

    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        act="relu",
        pna_degrees=None,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        norm_style="layer",
        local_norm_style="layer",
        norm_position="post",
        is_block_composed=False,
        is_ffn_block=True,
        num_dags=1,
        layer_args=None,
        log_attn_weights=False,
        d_inner=None,
        ffn_d_dinner=None,
        d_state=64,
        use_dag_decomposition=False,
    ):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        assert norm_style in ["layer", "batch", "none"]
        self.norm_style = norm_style
        assert norm_position in ["pre", "post"]
        self.norm_position = norm_position
        self.is_block_composed = is_block_composed
        self.is_ffn_block = is_ffn_block
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.d_inner = d_inner if d_inner is not None else dim_h
        self.ffn_d_dinner = ffn_d_dinner if ffn_d_dinner is not None else dim_h * 2
        self.d_state = d_state
        self.use_dag_decomposition = use_dag_decomposition

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in [
            "Transformer",
            "BiasedTransformer",
        ]:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == "None":
            self.local_model = None
        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == "GIN":
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(
                Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h)
            )
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == "GENConv":
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == "GINE":
            gin_nn = nn.Sequential(
                nn.BatchNorm1d(dim_h),
                # self.activation(),
                # nn.Dropout(dropout),
                nn.Linear(dim_h, dim_h),
            )
            # gin_nn = Linear_pyg(dim_h, dim_h)
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == "GAT":
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == "PNA":
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ["mean", "max", "sum"]
            scalers = ["identity"]
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=min(128, dim_h),
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == "CustomGatedGCN":
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                norm_style=local_norm_style,
                act=act,
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type
        if self.local_gnn_type is not None:
            self.local_normalizer = Normalizer(norm_style, dim_h)

        # Global attention transformer-style model.
        self.has_global = True
        if global_model_type == "None":
            self.self_attn = None
            self.vn = None
            self.has_global = False
        elif global_model_type in ["Transformer", "BiasedTransformer"]:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True
            )
        elif global_model_type == "Performer":
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False
            )
        elif global_model_type == "GraphSSDLayer":
            assert layer_args is not None, "need to have some layer args"
            if self.use_dag_decomposition:
                self.self_attn = DAGSSDLayer(
                    dim_in=dim_h,
                    dim_out=dim_h,
                    num_heads=num_heads,
                    d_inner=d_inner,
                    num_dags=num_dags,
                    dropout=dropout,
                    d_state=d_state,
                    layer_args=layer_args,
                )
            else:
                self.self_attn = GraphSSDLayer(
                    dim_in=dim_h,
                    dim_out=dim_h,
                    num_heads=num_heads,
                    d_inner=dim_h,
                    num_dags=num_dags,
                    dropout=dropout,
                    layer_args=layer_args,
                )

        else:
            raise ValueError(f"Unsupported global model: " f"{global_model_type}")
        self.global_model_type = global_model_type
        if self.global_model_type is not None:
            self.global_normalizer = Normalizer(norm_style, dim_h)

        # Feed Forward block.
        if self.is_ffn_block:
            self.ff_linear1 = nn.Linear(dim_h, self.ffn_d_dinner)
            self.ff_linear2 = nn.Linear(self.ffn_d_dinner, dim_h)
            self.act_fn_ff = self.activation()
            self.ffn_normalizer = Normalizer(norm_style, dim_h)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

        if (
            self.local_gnn_type != "CustomGatedGCN"
            or self.global_model_type != "GraphSSDLayer"
        ):
            warnings.warn(
                "This branch is not tested to work with other model combinations"
            )

    def forward(self, batch):
        h = batch.x

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            h_local_in = h

            if self.norm_position == "pre":
                h = self.local_normalizer(h, batch.batch)

            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == "CustomGatedGCN":
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(
                    Batch(
                        batch=batch.batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                        batch_edge=batch.batch_edge,
                    )
                )
                # GatedGCN does residual connection and dropout internally.
                h = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h = self.local_model(
                            h,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.pe_EquivStableLapPE,
                        )
                    else:
                        h = self.local_model(h, batch.edge_index, batch.edge_attr)
                else:
                    h = self.local_model(h, batch.edge_index)
                h = self.dropout_local(h)

            if self.norm_position == "post":
                h = self.local_normalizer(h_local_in + h, batch.batch)
            else:
                h = h + h_local_in

            if not self.is_block_composed:
                h_out_list.append(h)
                h = h_local_in

        # global information exchange.
        if self.has_global:
            h_global_in = h

            if self.norm_position == "pre":
                h = self.global_normalizer(h, batch.batch)
            h_dense, mask = to_dense_batch(h, batch.batch)

            if self.global_model_type == "Transformer":
                h = self._gl_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == "BiasedTransformer":
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h = self._gl_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == "Performer":
                h = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == "BigBird":
                h = self.self_attn(h_dense, attention_mask=mask)
            elif self.global_model_type == "GraphSSDLayer":
                h = self.self_attn(h, batch)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            if self.norm_position == "post":
                h = self.global_normalizer(h_global_in + h, batch.batch)
            else:
                h = h + h_global_in

            if not self.is_block_composed:
                h_out_list.append(h)
                h = h_local_in

        if not self.is_block_composed:
            h = sum(h_out_list)

        if self.is_ffn_block:
            h_ffn_in = h
            if self.norm_position == "pre":
                h = self.ffn_normalizer(h, batch.batch)
            h = self._ff_block(h)
            h = h + h_ffn_in
            if self.norm_position == "post":
                h = self.ffn_normalizer(h, batch.batch)

        batch.x = h
        return batch

    def _gl_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, "
            f"heads={self.num_heads}"
        )
        return s
