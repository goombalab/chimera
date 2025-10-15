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


class GraphSSD(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        num_heads=4,
        dt_multiplier=1,
        broadcast_bc_in_heads=True,
        norm_constraint="sum_leq",
        sub_constraint="max",
        combine_type="sum_node_edge",
        use_fast_inverse=True,
        # Constant Value Flags
        with_A=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        normalize_L=False,
        normalize_x=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_inner
        self.num_heads = num_heads
        self.dt_multiplier = dt_multiplier
        self.broadcast_bc_in_heads = broadcast_bc_in_heads
        self.norm_constraint = norm_constraint
        self.sub_constraint = sub_constraint
        self.combine_type = combine_type
        self.use_fast_inverse = use_fast_inverse

        # Fixed Flags
        self.with_A = with_A
        self.normalize_L = normalize_L
        self.normalize_x = normalize_x

        # self.tol = 1e-6
        self.tol = 0.05

        # self.num_heads = self.d_inner // headdim
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

        # Initialize A
        A = torch.ones(self.num_heads, dtype=torch.float32)
        self.A_log = torch.log(A).to(dtype=torch.float32)
        if self.with_A:
            self.A_log = nn.Parameter(self.A_log)
            self.A_log._no_reinit = True
            self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_reinit = True
        self.D._no_weight_decay = True

    def I_minus_A(self, A):
        I = (
            torch.eye(A.shape[-1], device=A.device, dtype=torch.float64)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        old_dtype = A.dtype
        A = A.to(torch.float64)
        if self.use_fast_inverse:
            L = DAGInverse.apply(torch.transpose(A, -1, -2), self.graph_diameter)
        else:
            L = torch.inverse(I - torch.transpose(A, -1, -2))
        L = L.to(old_dtype)
        return L

    def normalize_mat(self, norm_constraint, dt_mat, dt_dummy=None):
        """
        dt_mat: (b, n, l, l, d)
        dt_dummy: (b, n, l) # that is, define it for each node.
        """
        if norm_constraint in ["sum", "sum_leq"]:
            # NOTE: normalize the incoming edges. This should be fine.
            # normalizer = torch.sum(dt_mat, dim=-2)
            normalizer = torch.sum(dt_mat, dim=-1)
            if False:
                # NOTE: need to fix this
                normalizer = normalizer.masked_fill_(normalizer == 0, 1)
            if "leq" in self.norm_constraint:
                normalizer += dt_dummy
            if "max" in self.sub_constraint:
                normalizer = torch.max(torch.ones_like(normalizer), normalizer)
            if "min" in self.sub_constraint:
                normalizer = torch.mean(normalizer, dim=-1, keepdim=True)

            # dt_mat_normalized = dt_mat / (normalizer.unsqueeze(-2) + self.tol)
            dt_mat_normalized = dt_mat / (normalizer.unsqueeze(-1) + self.tol)
        else:
            raise ValueError("Invalid norm constraint")

        return dt_mat_normalized

    def get_L_matrix(self, edge_index, dt_A_node, dt_A_edge=None, data_batch=None):
        dt_dummy = None

        if self.combine_type in ["sum_node_edge"]:
            dt_s, dt_t = dt_A_node[0], dt_A_node[1]

            if dt_A_edge is not None:
                dt_A = (
                    dt_s[:, edge_index[0]] + dt_t[:, edge_index[1]] + dt_A_edge[0]
                ) / 3
            else:
                dt_A = (dt_s[:, edge_index[0]] + dt_t[:, edge_index[1]]) / 2
            dt_exp_A = torch.exp(dt_A)
            dt_exp_A = rearrange(dt_exp_A, "n l -> l n")
            dt_exp_A_mat = torch_geometric.utils.to_dense_adj(
                edge_index, edge_attr=dt_exp_A, batch=data_batch
            )
            dt_exp_A_mat = rearrange(dt_exp_A_mat, "b l1 l2 h -> b h l1 l2")

            if "leq" in self.norm_constraint:
                dt_dummy = torch.exp(dt_A_node[2])
                dt_dummy = rearrange(dt_dummy, "n l -> l n")
                dt_dummy, _ = torch_geometric.utils.to_dense_batch(
                    dt_dummy, batch=data_batch
                )
                dt_dummy = rearrange(dt_dummy, "b l1 h -> b h l1")

            A_mat = self.normalize_mat(self.norm_constraint, dt_exp_A_mat, dt_dummy)
            L = self.I_minus_A(A_mat)
            # print(L)
            if self.normalize_L:
                L = self.normalize_mat("sum", L)
            return L
        else:
            ValueError("Invalid Combine Type")

    def multiply_dt_w_A(self, dt):
        dt = dt + self.dt_bias.unsqueeze(0).unsqueeze(-1)
        dt = F.softplus(dt)
        A = -torch.exp(self.A_log.unsqueeze(0).unsqueeze(-1)).to(dt.device)
        dt_A = A * dt
        return dt_A

    def get_degree_normalize(self, x, edge_index, data_batch):
        degree = torch_geometric.utils.degree(edge_index[1], x.shape[0])
        deg_dense, _ = torch_geometric.utils.to_dense_batch(degree, data_batch)
        deg_dense_inv = deg_dense.pow(-1)
        deg_dense_inv = deg_dense_inv.masked_fill_(deg_dense_inv == float("inf"), 1)
        return deg_dense_inv

    def forward(
        self,
        x,
        BC,
        dt,
        dt_edge,
        edge_index,
        data_batch,
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

        B, C = BC
        B_dense, _ = torch_geometric.utils.to_dense_batch(B, data_batch)
        C_dense, _ = torch_geometric.utils.to_dense_batch(C, data_batch)
        # Compute normalized matrix mixer outputs:
        if self.broadcast_bc_in_heads:
            B_dense = B_dense.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            C_dense = C_dense.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        else:
            B_dense = rearrange(B_dense, "b l (n d) -> b n l d", n=self.num_heads)
            C_dense = rearrange(C_dense, "b l (n d) -> b n l d", n=self.num_heads)

        dt = rearrange(dt, "l (n c) -> c n l", n=self.num_heads)
        dt_node, dt_self = dt[:-1], dt[-1]  # leave the last one for dt_self.

        dt_self = rearrange(dt_self, "n l -> l n")
        dt_self_dense, _ = torch_geometric.utils.to_dense_batch(dt_self, data_batch)
        dt_self_dense = rearrange(dt_self_dense, "b l n -> b n l")

        dt_A_node = self.multiply_dt_w_A(dt_node)
        if dt_edge is not None:
            dt_edge = rearrange(dt_edge, "l (n c) -> c n l", n=self.num_heads)
            dt_A_edge = self.multiply_dt_w_A(dt_edge)
        else:
            dt_A_edge = None

        L = self.get_L_matrix(edge_index, dt_A_node, dt_A_edge, data_batch)

        # L_normalized = L * dt_self_dense.unsqueeze(-1)
        L_normalized = L * dt_self_dense.unsqueeze(-2)

        if self.normalize_x:
            deg_inv = self.get_degree_normalize(x, edge_index, data_batch)
            x_rearranged = x_rearranged * deg_inv.unsqueeze(1).unsqueeze(-1)

        y = torch.einsum(
            "b n l d, b n l t, b n t d, b n t h -> b n l h",
            C_dense,
            L_normalized,
            B_dense,
            x_rearranged,
        )
        y = y + x_rearranged * self.D.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        y = rearrange(y, "b n l h -> b l (h n)")
        y = y[mask]
        return y


class GraphSSDLayer(nn.Module):
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

        # NEW FLAGS
        self.is_share_BC_dags = layer_args.is_share_BC_dags  # share BC across heads
        self.is_edge_dt = layer_args.is_edge_dt  # use edge features for dt
        self.conv_type = layer_args.conv_type
        assert self.conv_type in ["identity", "gcn", "gin", "gine", "depthwise_local"]

        # OLD Flags
        self.norm_constraint = layer_args.norm_constraint
        self.sub_constraint = layer_args.sub_constraint
        self.combine_type = layer_args.combine_type
        self.broadcast_bc_in_heads = layer_args.broadcast_bc_in_heads
        self.dt_multiplier = layer_args.dt_multiplier
        self.use_fast_inverse = layer_args.use_fast_inverse
        self.dt_min = layer_args.dt_min
        self.dt_max = layer_args.dt_max

        self.graph_layer = GraphSSD(
            d_model=self.d_model,
            d_inner=self.d_inner,
            num_heads=self.num_heads,
            dt_multiplier=self.dt_multiplier,
            broadcast_bc_in_heads=self.broadcast_bc_in_heads,
            norm_constraint=self.norm_constraint,
            sub_constraint=self.sub_constraint,
            combine_type=self.combine_type,
            use_fast_inverse=self.use_fast_inverse,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
        )

        self.num_bc = 1 if self.broadcast_bc_in_heads else self.num_heads
        self.add_dummy_dt = 1 if "_leq" in self.graph_layer.norm_constraint else 0
        self.is_sum_constr = "sum" in self.graph_layer.norm_constraint
        self.num_dt = (2 + self.is_sum_constr + self.add_dummy_dt) * self.num_heads
        d_in_proj = (
            self.d_inner + self.d_inner + 2 * self.num_bc * self.d_state + self.num_dt
        )

        # Create projection layers
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        if self.is_edge_dt:
            self.edge_proj = nn.Linear(self.d_model, self.num_heads, bias=False)
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
        self.approx_factor = int(layer_args.approx_factor)

        # print(f"GraphSSDLayer for GPS Layer initialized on rank {dist.get_rank()}")

    def forward(self, x, batch):
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        data_batch = getattr(batch, "batch", None)

        assert (
            edge_index is not None and edge_attr is not None and data_batch is not None
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
        diameter = torch.max(torch.bincount(data_batch)).item()//self.approx_factor
        x = self.graph_layer(
            x=x,
            BC=[B, C],
            dt=dt,
            dt_edge=dt_edge,
            edge_index=edge_index,
            data_batch=data_batch,
            diameter=diameter,
        )
        x = self.norm(x + x_in, z)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
