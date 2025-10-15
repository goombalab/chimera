import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pygnn
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import MessagePassing

import networkx as nx


class DAGInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, n):
        j = 2 ** math.ceil(math.log2(n))

        # |A|: (b n l l)
        last_A_j = A
        I_matrix = (
            torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        I_minus_A_inv = I_matrix + A

        # Compute (I+A)(I+A^2)(I+A^4)...(I+A^j)
        # TODO: Do this in log(j) steps, but for j = 32, O(j) is fine
        for _ in range(2, int(math.log2(j)) + 1):
            last_A_j = torch.matmul(last_A_j, last_A_j)
            I_minus_A_inv = torch.matmul(I_minus_A_inv, I_matrix + last_A_j)

        ctx.save_for_backward(I_minus_A_inv)
        return I_minus_A_inv

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved tensors from the forward pass
        I_minus_A_inv = ctx.saved_tensors[0]

        # Compute the gradient with respect to A
        grad_A = torch.matmul(
            I_minus_A_inv.transpose(-2, -1),
            torch.matmul(grad_output, I_minus_A_inv.transpose(-2, -1)),
        )

        return grad_A, None


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, "id", "identity", "linear", "none"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation)
        )


class LocalConv1D(MessagePassing):
    def __init__(self, dim_in):
        super().__init__(aggr="add")
        self.conv_neighbors = nn.Conv1d(
            in_channels=dim_in, out_channels=dim_in, groups=dim_in, kernel_size=1
        )
        self.conv_self = nn.Conv1d(
            in_channels=dim_in,
            out_channels=dim_in,
            groups=dim_in,
            kernel_size=1,
        )

    def forward(self, x, edge_index):
        # just to be sure, pretty sure there aren't self loops left.
        edge_index_self_removed, _ = torch_geometric.utils.remove_self_loops(edge_index)

        x_neighbors = self.conv_neighbors(x.unsqueeze(-1)).squeeze()
        x_neighbors = self.propagate(edge_index_self_removed, x=x_neighbors)

        x_self = self.conv_self(x.unsqueeze(-1)).squeeze()

        return x_self + x_neighbors


class DAGSSD(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        num_heads=4,
        dt_multiplier=1,
        broadcast_bc_in_heads=True,
        use_fast_inverse=True,
        num_dags=1,
        is_share_BC_dags=True,
        # Constant Value Flags
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_inner
        self.num_heads = num_heads
        self.dt_multiplier = dt_multiplier
        self.broadcast_bc_in_heads = broadcast_bc_in_heads
        self.use_fast_inverse = use_fast_inverse
        self.num_dags = num_dags
        self.is_share_BC_dags = is_share_BC_dags
        self.tol = 1e-6
        self.headdim = self.d_inner // self.num_heads
        assert self.d_inner % self.headdim == 0

        # Initialize log dt bias
        dt_shape = (self.num_heads,)
        dt = torch.exp(
            torch.rand(
                dt_shape,
            )
            * (
                math.log(self.dt_multiplier * dt_max)
                - math.log(self.dt_multiplier * dt_min)
            )
            + math.log(self.dt_multiplier * dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_reinit = True
        self.dt_bias._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_reinit = True
        self.D._no_weight_decay = True

    def I_minus_A_inv(self, A):
        I = (
            torch.eye(A.shape[-1], device=A.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if self.use_fast_inverse:
            L = DAGInverse.apply(A.to(torch.float32), self.graph_diameter)
        else:
            L = torch.inverse(I - A)
        return L

    def normalize_mat(
        self,
        dt_mat,
        dt_sum,
        data_batch,
        dag_masks,
        edge_index,
    ):
        """
        dt_mat: (b, n, l, l, d)
        dt_dummy: (b, n, l) # that is, define it for each node.
        """
        num_nodes, num_edges = (
            data_batch.shape[0],
            edge_index.shape[-1],
        )
        # Get the number of incident edges for each node
        target_nodes = edge_index[-1]
        edge_count = torch.zeros((self.num_dags, num_nodes), device=dt_mat.device)
        ones = torch.ones(self.num_dags, num_edges, device=dt_mat.device) * dag_masks
        edge_count.scatter_add_(
            -1, target_nodes.unsqueeze(0).expand(self.num_dags, -1), ones
        )  # (g, L)
        edge_count = rearrange(edge_count, "g l -> l g")
        edge_count, _ = torch_geometric.utils.to_dense_batch(
            edge_count, data_batch
        )  # (b, l1, g)
        sqrt_edge_count_outer = torch.sqrt(
            torch.einsum("b l g, b t g -> b g l t", edge_count, edge_count)
        )
        sqrt_edge_count_outer[sqrt_edge_count_outer < 1] = 1  # Avoid division by zero

        # Get sum up dt_sum for each incident edge
        dt_normalizer = torch.zeros(
            (self.num_dags, self.num_heads, num_nodes), device=dt_mat.device
        )
        dt_sum_masked = rearrange(
            dt_sum, "l g n -> g n l", g=self.num_dags, n=self.num_heads
        )
        dt_normalizer.scatter_add_(
            -1,
            target_nodes.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_dags, self.num_heads, -1),
            dt_sum_masked,
        )  # (g, n, L)

        dt_normalizer = rearrange(dt_normalizer, "g n l -> l (g n)")
        dt_normalizer, _ = torch_geometric.utils.to_dense_batch(
            dt_normalizer, data_batch
        )  # (b, l1, (g, n))
        dt_normalizer = rearrange(
            dt_normalizer,
            "b l1 (g n) -> b g n l1",
            g=self.num_dags,
            n=self.num_heads,
        )

        # Normalize dt_mat (b g n l1 l2)
        dt_mat = dt_mat / sqrt_edge_count_outer.unsqueeze(2).expand(
            -1, -1, self.num_heads, -1, -1
        )

        return dt_mat, dt_normalizer

    def get_L_matrix(self, edge_index, dt_node, dt_edge, data_batch, dag_masks):
        dt_s, dt_t = dt_node[0], dt_node[1]

        if dt_edge is not None:
            dt_sum = (
                dt_s[:, :, edge_index[0]] + dt_t[:, :, edge_index[1]] + dt_edge
            ) / math.sqrt(3)
        else:
            dt_sum = (
                dt_s[:, :, edge_index[0]] + dt_t[:, :, edge_index[1]]
            ) / math.sqrt(2)

        dt_sum = rearrange(dt_sum, "g n l -> l g n")
        dt_exp = torch.exp(-dt_sum)

        # Mask out the DAG Edges
        if dag_masks is not None:
            mask = rearrange(dag_masks, "g l -> l g")
            mask = mask.unsqueeze(-1).expand(-1, -1, self.num_heads)  # (l, g, n)
            dt_sum = dt_sum * mask
            dt_exp = dt_exp * mask

        dt_exp_ng = rearrange(dt_exp, "l g n -> l (g n)")
        dt_exp_mat = torch_geometric.utils.to_dense_adj(
            edge_index, edge_attr=dt_exp_ng, batch=data_batch
        )
        dt_exp_mat = rearrange(
            dt_exp_mat,
            "b l1 l2 (g n) -> b g n l2 l1",
            g=self.num_dags,
            n=self.num_heads,
        )  # (NOTE) the transpose is important

        dt_exp_mat_normalized, dt_normalizer = self.normalize_mat(
            dt_mat=dt_exp_mat,
            dt_sum=dt_sum,
            data_batch=data_batch,
            dag_masks=dag_masks,
            edge_index=edge_index,
        )
        dt_exp_mat_normalized = rearrange(
            dt_exp_mat_normalized, "b g n l1 l2 -> b (g n) l1 l2"
        )

        L = self.I_minus_A_inv(dt_exp_mat_normalized)
        L = rearrange(
            L,
            "b (g n) l1 l2 -> b g n l1 l2",
            g=self.num_dags,
            n=self.num_heads,
        )
        L = L * dt_normalizer.unsqueeze(-1).expand(-1, -1, -1, -1, L.shape[-1])
        return L

    def get_dt_softplus(self, dt):
        dt = dt + self.dt_bias.unsqueeze(0).unsqueeze(-1)
        dt = F.softplus(dt)
        return dt

    def check_if_dag(self, edge_index):
        # Convert `edge_index` to a list of tuples (edges)
        edges = (
            edge_index.detach().cpu().t().tolist()
        )  # Assuming `edge_index` is a 2xN tensor

        # Create a directed graph
        G = nx.DiGraph()
        G.add_edges_from(edges)

        # Check if the graph is a DAG
        is_dag = nx.is_directed_acyclic_graph(G)

        return is_dag

    def forward(
        self,
        x,
        BC,
        dt,
        dt_edge,
        edge_index,
        data_batch,
        dag_masks,
        diameter,
    ):
        """
        x: (num_nodes, d_input)
        BC: (num_nodes, n_heads, d_state)
        BC: (num_nodes, n_heads, d_state)
        dt: (num_nodes, num_dts, n_heads)
        """
        self.graph_diameter = diameter
        x_dense, mask = torch_geometric.utils.to_dense_batch(x, data_batch)
        x_rearranged = rearrange(x_dense, "b l (n h) -> b n l h", n=self.num_heads)
        x_rearranged = x_rearranged.unsqueeze(1).expand(-1, self.num_dags, -1, -1, -1)

        B, C = BC
        B_dense, _ = torch_geometric.utils.to_dense_batch(B, data_batch)
        C_dense, _ = torch_geometric.utils.to_dense_batch(C, data_batch)

        # for dag_idx in range(self.num_dags):
        #     assert self.check_if_dag(edge_index[:, dag_masks[dag_idx]])

        B_dense = rearrange(
            B_dense,
            "b l (g n d) -> b g n l d",
            g=1 if self.is_share_BC_dags else self.num_dags,
            n=1 if self.broadcast_bc_in_heads else self.num_heads,
        )
        C_dense = rearrange(
            C_dense,
            "b l (g n d) -> b g n l d",
            g=1 if self.is_share_BC_dags else self.num_dags,
            n=1 if self.broadcast_bc_in_heads else self.num_heads,
        )

        if self.broadcast_bc_in_heads:
            B_dense = B_dense.repeat(1, 1, self.num_heads, 1, 1)
            C_dense = C_dense.repeat(1, 1, self.num_heads, 1, 1)

        if self.is_share_BC_dags:
            B_dense = B_dense.repeat(1, self.num_dags, 1, 1, 1)
            C_dense = C_dense.repeat(1, self.num_dags, 1, 1, 1)

        dt_node = rearrange(
            dt, "l (g n c) -> c g n l", g=self.num_dags, n=self.num_heads
        )
        dt_node = self.get_dt_softplus(dt_node)

        if dt_edge is not None:
            dt_edge = rearrange(dt_edge, "l (g n) -> g n l", n=self.num_heads)
            dt_edge = self.get_dt_softplus(dt_edge)
        else:
            dt_edge = None

        L_normalized = self.get_L_matrix(
            edge_index=edge_index,
            dt_node=dt_node,
            dt_edge=dt_edge,
            data_batch=data_batch,
            dag_masks=dag_masks,
        )

        y = torch.einsum(
            "b g n l d, b g n l t, b g n t d, b g n t h -> b g n l h",
            C_dense,
            L_normalized,
            B_dense,
            x_rearranged,
        )
        y = y + x_rearranged * self.D.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        y = rearrange(y, "b g n l h -> b l (g n h)")
        y = y[mask]
        y = rearrange(y, "l (g n h) -> g l (n h)", g=self.num_dags, n=self.num_heads)
        y = torch.sum(y, dim=0, keepdim=False) / math.sqrt(self.num_dags)
        return y


class DAGSSDLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        layer_args,
        num_dags=1,
        num_heads=4,
        d_inner=128,
        d_state=64,
        activation="swish",
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = dim_in
        self.d_out = dim_out

        assert activation in ["swish", "silu"]
        self.act = Activation(activation)
        self.d_inner = d_inner
        self.d_state = d_state
        self.num_heads = num_heads
        self.num_dags = num_dags

        # NEW FLAGS
        self.is_share_BC_dags = layer_args.is_share_BC_dags  # share BC across heads
        self.is_edge_dt = layer_args.is_edge_dt  # use edge features for dt
        self.conv_type = layer_args.conv_type
        assert self.conv_type in ["identity", "gcn", "gin", "gine", "depthwise_local"]

        # OLD Flags
        self.broadcast_bc_in_heads = layer_args.broadcast_bc_in_heads
        self.dt_multiplier = layer_args.dt_multiplier
        self.use_fast_inverse = layer_args.use_fast_inverse

        self.graph_layer = DAGSSD(
            d_model=self.d_model,
            d_inner=self.d_inner,
            num_heads=self.num_heads,
            dt_multiplier=self.dt_multiplier,
            broadcast_bc_in_heads=self.broadcast_bc_in_heads,
            is_share_BC_dags=self.is_share_BC_dags,
            use_fast_inverse=self.use_fast_inverse,
            num_dags=self.num_dags,
        )

        self.num_bc = (1 if self.is_share_BC_dags else self.num_dags) * (
            1 if self.broadcast_bc_in_heads else self.num_heads
        )
        self.num_dt = 2 * self.num_dags * self.num_heads
        d_in_proj = (
            self.d_inner + self.d_inner + 2 * self.num_bc * self.d_state + self.num_dt
        )

        # Create projection layers
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        if self.is_edge_dt:
            self.edge_proj = nn.Linear(
                self.d_model, self.num_dags * self.num_heads, bias=False
            )
        self.out_proj = nn.Linear(self.d_inner, self.d_out, bias=False)

        # Create a graphconv layer
        conv_dim = self.d_inner + self.d_state * 2 * self.num_bc
        if self.conv_type.lower() == "gcn":
            self.local_conv_layer = pygnn.GCNConv(conv_dim, conv_dim)
        elif self.conv_type.lower() == "gin":
            gin_nn = nn.Sequential(
                Linear_pyg(conv_dim, conv_dim),
                Activation("relu"),
                Linear_pyg(conv_dim, conv_dim),
            )
            self.local_conv_layer = pygnn.GINCONV(gin_nn)
        elif self.conv_type.lower() == "gine":  # unsupported.
            # NOTE: Gine is not supported right now. need to figure out how to handle edge features
            gin_nn = nn.Sequential(
                nn.BatchNorm1d(conv_dim), nn.Linear(conv_dim, conv_dim)
            )
            self.local_conv_layer = pygnn.GINEConv(gin_nn)
        elif self.conv_type.lower() == "depthwise_local":
            self.local_conv_layer = LocalConv1D(dim_in=conv_dim)
        else:
            self.local_conv_layer = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = RMSNorm(self.d_inner, eps=1e-5, norm_before_gate=False)

        print("GraphSSDLayer for GPS Layer initialized")

    def forward(self, x, batch):
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        dag_masks = getattr(batch, "dag_masks", None)
        data_batch = getattr(batch, "batch", None)

        assert (
            edge_index is not None
            and edge_attr is not None
            and dag_masks is not None
            and data_batch is not None
        )

        # Project the node features to get z,x,B,C,dt
        zxBCdt = self.in_proj(x)

        z, xBC, dt = torch.split(
            zxBCdt,
            [
                self.d_inner,
                self.d_inner + (2 * self.d_state * self.num_bc),
                self.num_dt,
            ],
            dim=-1,
        )
        if self.is_edge_dt:
            dt_edge = self.edge_proj(edge_attr)
        else:
            dt_edge = None
        zero_hack = torch.sum(edge_attr * 0)

        if self.conv_type in ["identity"]:
            xBC = self.act(self.local_conv_layer(xBC))
        else:
            xBC = self.act(self.local_conv_layer(xBC, edge_index))

        x, BC = torch.split(
            xBC, [self.d_inner, 2 * (self.d_state * self.num_bc)], dim=-1
        )
        B, C = torch.split(
            BC, [self.num_bc * self.d_state, self.num_bc * self.d_state], dim=-1
        )

        x_in = x
        diameter = torch.max(torch.bincount(data_batch)).item()
        x = self.graph_layer(
            x=x,
            BC=[B, C],
            dt=dt,
            dt_edge=dt_edge,
            edge_index=edge_index,
            data_batch=data_batch,
            dag_masks=dag_masks,
            diameter=diameter,
        )
        x = self.norm(x + x_in, z)
        x = self.dropout(x)
        x = self.out_proj(x) + zero_hack
        return x
