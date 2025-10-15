import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head("mlp_graph")
class MLPGraphHead(nn.Module):
    """
    MLP prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        if cfg.model.graph_pooling != "node_ensemble":
            self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
            self.node_ensemble = False
        else:
            self.pooling_fun = register.pooling_dict["mean"]
            self.node_ensemble = True

        self.add_layer_norm = cfg.gt.gnn_layernorm
        self.add_residual = cfg.gt.gnn_residual
        dropout = cfg.gnn.dropout
        L = cfg.gnn.layers_post_mp

        self.mlps = nn.ModuleList()
        if self.add_layer_norm:
            self.ln_s = nn.ModuleList()

        for _ in range(L - 1):
            layers = []
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_in, bias=True))
            layers.append(register.act_dict[cfg.gnn.act]())
            self.mlps.append(nn.Sequential(*layers))
            if self.add_layer_norm:
                self.ln_s.append(nn.LayerNorm(dim_in))

        layers = []
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim_in, dim_out, bias=True))
        self.mlps.append(nn.Sequential(*layers))

    def _scale_and_shift(self, x):
        return x

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        if self.node_ensemble:
            x = batch.x
        else:
            x = self.pooling_fun(batch.x, batch.batch)

        for idx in range(len(self.mlps) - 1):
            if self.add_layer_norm:
                norm_x = self.ln_s[idx](x)
            else:
                norm_x = x

            if self.add_residual:
                x = self.mlps[idx](norm_x) + x
            else:
                x = self.mlps[idx](norm_x)

        x = self.mlps[-1](x)

        y = self._scale_and_shift(x)

        if self.node_ensemble:
            y_graph = self.pooling_fun(y, batch.batch)
            batch.graph_feature = y_graph

            _, label = self._apply_index(batch)
            if self.training:
                return y, label[batch.batch]
            else:
                return y_graph, label

        else:
            batch.graph_feature = y
            pred, label = self._apply_index(batch)
            return pred, label


@register_head("mlp_graph_pcqm4m")
class MLPGraphHeadPCQM4M(MLPGraphHead):

    def _scale_and_shift(self, x):
        return (x * 1.1623) + 5.6896


@register_head("mlp_graph_zinc")
class MLPGraphHeadZINC(MLPGraphHead):

    def _scale_and_shift(self, x):
        return (x * 2.0109) + 0.0153
