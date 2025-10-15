import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
import sys 
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

class DAGInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, n):
        j = 2 ** math.ceil(math.log2(n))
        
        #|A|: (b n l l)
        last_A_j = A
        I_matrix =  torch.eye(A.size(-1), dtype=A.dtype, device=A.device).unsqueeze(0).unsqueeze(0)
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
        grad_A = torch.matmul(I_minus_A_inv.transpose(-2, -1), torch.matmul(grad_output, I_minus_A_inv.transpose(-2, -1)))
        
        return grad_A, None

class Chimera(nn.Module):

    def __init__(
        self,
        d_model,
        qk_dim=64,
        expand_factor=2.0,
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
        norm_sqrt_mul_factor=1.0, # < 1
        # other configs
        dt_min=0.001,
        dt_max=0.1,
        dt_init='random',
        dt_init_scale=1.0,
        dt_init_floor=1e-4,
        device=None,
        dtype=None,
        image_height=14, #Need to support dynamic image height
        image_width=14, #Need to support dynamic image width
    ):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.expand_factor = expand_factor
        self.headdim = headdim
        # self.max_seq_len = max_seq_len

        # graph mamba flags
        self.unified_view = unified_view
        self.include_headnodes = include_headnodes
        self.debug_use_get_A_dpr = debug_use_get_A_dpr
        self.debug_store_mm = debug_store_mm
        self.share_BC = share_BC
        self.share_BC_for_two_graphs = share_BC_for_two_graphs
        self.share_dt_for_two_graphs = share_dt_for_two_graphs
        self.share_BC_for_two_graphs_mode = share_BC_for_two_graphs_mode
        self.use_fast_inverse = use_fast_inverse
        self.dt_min = dt_min*dt_min_max_factor
        self.dt_max = dt_max*dt_min_max_factor
        self.dt_self_min = dt_min*dt_self_min_max_factor
        self.dt_self_max = dt_max*dt_self_min_max_factor
        self.normalization_mode = normalization_mode
        self.norm_sqrt_mul_factor = norm_sqrt_mul_factor

        self.d_inner = round(self.expand_factor * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.num_heads = self.d_inner // self.headdim
        self.std_dev = 1.0 # This will be per method parameter
        self.tol = 1e-6
        self.inverse_factor=0.95
        self.image_height = image_height
        self.image_width = image_width

        # Initialize log dt bias
        dt_shape = (self.num_heads,)
        dt = torch.exp(
            torch.rand(dt_shape,) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_reinit = True
        self.dt_bias._no_weight_decay = True

        if self.normalization_mode == "dt_self":
            dt_self = torch.exp(
                torch.rand(dt_shape,) * (math.log(self.dt_self_max) - math.log(self.dt_self_min)) + math.log(self.dt_self_min)
            )
            dt_self = torch.clamp(dt_self, min=dt_init_floor)
            inv_dt_self = dt_self + torch.log(-torch.expm1(-dt_self))
            self.dt_self_bias = nn.Parameter(inv_dt_self)
            self.dt_self_bias._no_reinit = True
            self.dt_self_bias._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_heads, device=device))
        self.D._no_reinit = True
        self.D._no_weight_decay = True

        self.unused_dt_mask = self.get_unused_dt_mask().unsqueeze(2).unsqueeze(2)

        if self.debug_store_mm:
            self.epoch=0

    def get_unused_dt_mask(self):
        # Shape of dt_rearranged: [4, 2, batch, num_heads, i, j]
        # Must be used in the order [0]top_left, [1]top_right, [2]bottom_left, [3]bottom_right,
        # where each is followed by [0]left-right, [1]top-bottom

        # Visualize a Rectangle with 4 corners
        # 0 --(a)--- 1
        # |          |
        #(d)        (b) 
        # |          |
        # 2 --(c)--- 3
        
        dt_mask = torch.ones(
            (4, 2, self.image_height, self.image_width),
            requires_grad=False)

        # Corners and Edges: 
        # For (a) remove [0][1], and [1][1]
        dt_mask[0][1][0, :] = 0
        dt_mask[1][1][0, :] = 0

        # For (b) remove [1][0], and [3][0]
        dt_mask[1][0][:, -1] = 0
        dt_mask[3][0][:, -1] = 0

        # For (c) remove [2][1], and [3][1]
        dt_mask[2][1][-1, :] = 0
        dt_mask[3][1][-1, :] = 0

        # For (d) remove [0][0], and [2][0]
        dt_mask[0][0][:, 0] = 0
        dt_mask[2][0][:, 0] = 0

        idxs = [i for i in range(4) if self.include_headnodes[i] == '1']
        return dt_mask[idxs]
    
    # Returns DAG mixers
    def get_normalized_dag_mixer_depr(
        self, 
        expdt, 
        head_node_type, 
        device,
        is_normalize=True):

        # Shape of dt_rearranged: [2, batch, num_heads, i, j]
        # where (i -> rows,j -> column)
        # 2: corresponds to left_right edges and top_bottom edges
        # |head_node_type|: values top_left, top_right, bottom_left, bottom_right

        # Exponentiate all dts
        height, width = expdt.shape[-2:]

        # num incident edges on each node
        num_incident_edges = 2*torch.ones((height, width), device=device, requires_grad=False)

        # left_right matrix ----------
        # left_right node indices
        lr_node_indices = torch.arange(height*width).reshape(height, width).to(device)

        lr_left_nodes = lr_node_indices[:, :-1].flatten()
        lr_right_nodes = lr_node_indices[:, 1:].flatten()
        lr_dt = expdt[0] # (b n i j)
        
        if head_node_type == 'top_left' or head_node_type == 'bottom_left':
            lr_edge_index = torch.stack([lr_left_nodes, lr_right_nodes], dim=0)
            lr_edge_attr = lr_dt[:,:,:,1:] # ensures target data dependency
            lr_edge_attr = rearrange(lr_edge_attr, "b n i j -> (i j) b n")
            num_incident_edges[:,0] = num_incident_edges[:,0] - 1

        elif head_node_type == 'top_right' or head_node_type == 'bottom_right':
            lr_edge_index = torch.stack([lr_right_nodes, lr_left_nodes], dim=0)
            lr_edge_attr = lr_dt[:,:,:,:-1] # ensures target data dependency
            lr_edge_attr = rearrange(lr_edge_attr, "b n i j -> (i j) b n")
            num_incident_edges[:,-1] = num_incident_edges[:,-1] - 1

        else:
            raise ValueError(f"Invalid head node: {head_node_type}")

        # top_bottom matrix ----------------
        # top_bottom node indices
        tb_node_indices = torch.arange(height*width).reshape(height, width).to(device)

        tb_top_nodes = tb_node_indices[:-1, :].flatten()
        tb_bottom_nodes = tb_node_indices[1:, :].flatten()
        tb_dt = expdt[1] # (b n i j)

        if head_node_type == 'top_left' or head_node_type == 'top_right':
            tb_edge_index = torch.stack([tb_top_nodes, tb_bottom_nodes], dim=0)
            tb_edge_attr = tb_dt[:,:,1:,:]
            tb_edge_attr = rearrange(tb_edge_attr, "b n i j -> (i j) b n")
            num_incident_edges[0,:] = num_incident_edges[0,:] - 1
        
        elif head_node_type == 'bottom_left' or head_node_type == 'bottom_right':
            tb_edge_index = torch.stack([tb_bottom_nodes, tb_top_nodes], dim=0)
            tb_edge_attr = tb_dt[:,:,:-1,:]
            tb_edge_attr = rearrange(tb_edge_attr, "b n i j -> (i j) b n")
            num_incident_edges[-1,:] = num_incident_edges[-1,:] - 1
        
        else:
            raise ValueError(f"Invalid head node: {head_node_type}")
        
        graph_data = Data(
            edge_index=torch.concatenate((lr_edge_index, tb_edge_index), dim=1), 
            edge_attr=torch.concatenate((lr_edge_attr, tb_edge_attr), dim=0))
        
        adjacency_matrix = to_dense_adj(
            graph_data.edge_index, edge_attr=graph_data.edge_attr)
        adjacency_matrix = rearrange(adjacency_matrix[0], "l t b n -> b n l t")

        # Transpose the last two dimensions, and make it an incoming edges matrix
        adjacency_matrix = adjacency_matrix.transpose(-1, -2)
         # Divide each row by the number of non-zero values
        num_incident_edges[num_incident_edges < self.tol] = 1
        if is_normalize:
            final_adjacency_matrix = adjacency_matrix / torch.sqrt(
                rearrange(num_incident_edges, "i j -> (i j)")).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        else:
            final_adjacency_matrix = adjacency_matrix

        return final_adjacency_matrix, num_incident_edges

    # Returns DAG mixers
    def get_normalized_dag_mixer(
            self, 
            expdt,
            head_node_type,
            device,
            is_normalize=True):
        # Shape of dt_rearranged: [2, batch, num_heads, i, j]
        # where (i -> rows,j -> column)
        # 2: corresponds to left_right edges and top_bottom edges
        # |head_node_type|: values top_left, top_right, bottom_left, bottom_right

        batch, num_heads, height, width = expdt[0].shape
        seq_len = height * width

        # Matrix of all zeros (b n i j l)
        adjacency_matrix = torch.zeros(
            (batch, num_heads, height*width, height*width), device=device)
        # num incident edges on each node
        num_incident_edges = 2*torch.ones((height, width), device=device, requires_grad=False)

        # left_right matrix ----------
        index_set_2d = torch.arange(0, seq_len).reshape(height, width).to(device)

        # left_right node dt values
        lr_dt = expdt[0] # (b n i j)

        if head_node_type == 'top_left' or head_node_type == 'bottom_left':
            left_set = index_set_2d[:, :-1].flatten()
            right_set = index_set_2d[:, 1:].flatten()
            value_set = rearrange(lr_dt, "b n i j -> b n (i j)")
            adjacency_matrix[:, :, left_set, right_set] = value_set[:, :, right_set]

            num_incident_edges[:,0] = num_incident_edges[:,0] - 1

        elif head_node_type == 'top_right' or head_node_type == 'bottom_right':
            left_set = index_set_2d[:, 1:].flatten()
            right_set = index_set_2d[:, :-1].flatten()
            value_set = rearrange(lr_dt, "b n i j -> b n (i j)")
            adjacency_matrix[:, :, left_set, right_set] = value_set[:, :, right_set]

            num_incident_edges[:,-1] = num_incident_edges[:,-1] - 1

        else:
            raise ValueError(f"Invalid head node: {head_node_type}")

        # # top_bottom matrix ----------------
        # top_bottom node dt values
        tb_dt = expdt[1] # (b n i j)

        if head_node_type == 'top_left' or head_node_type == 'top_right':
            left_set = index_set_2d[:-1, :].flatten()
            right_set = index_set_2d[1:, :].flatten()
            value_set = rearrange(tb_dt, "b n i j -> b n (i j)")
            adjacency_matrix[:, :, left_set, right_set] = value_set[:, :, right_set]

            num_incident_edges[0,:] = num_incident_edges[0,:] - 1

        elif head_node_type == 'bottom_left' or head_node_type == 'bottom_right':
            left_set = index_set_2d[1:, :].flatten()
            right_set = index_set_2d[:-1, :].flatten()
            value_set = rearrange(tb_dt, "b n i j -> b n (i j)")
            adjacency_matrix[:, :, left_set, right_set] = value_set[:, :, right_set]

            num_incident_edges[-1,:] = num_incident_edges[-1,:] - 1
        else:
            raise ValueError(f"Invalid head node: {head_node_type}")

        # Transpose the last two dimensions, and make it an incoming edges matrix
        adjacency_matrix = adjacency_matrix.transpose(-1, -2)

        # Divide each row by the number of non-zero values
        num_incident_edges[num_incident_edges < self.tol] = 1

        if is_normalize:
            final_adjacency_matrix = adjacency_matrix / torch.sqrt(
                rearrange(num_incident_edges, "i j -> (i j)")).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        else:
            final_adjacency_matrix = adjacency_matrix

        return final_adjacency_matrix, num_incident_edges


    def forward(self, hidden_states, wgt_params_data, bc, dt_self=None):
        # Rearrange hidden states to shape [batch, n_heads, length, qk_dim]

        assert self.image_height*self.image_width == hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        num_matrix_mixers = sum(map(int, self.include_headnodes))
        hidden_states_rearranged = rearrange(
            hidden_states, 
            'b l (n h) -> b n l h', 
            n=self.num_heads)
        device = hidden_states.device
        self.unused_dt_mask = self.unused_dt_mask.to(device) #(num_matrix_mixers,2,i,j)

        if self.debug_use_get_A_dpr:
            get_normalized_dag_mixer = self.get_normalized_dag_mixer_depr
        else:
            get_normalized_dag_mixer = self.get_normalized_dag_mixer

        if self.debug_store_mm and str(device) == 'cuda:0':
            self.epoch = self.epoch + 1

        # Rearrange dt
        # (i -> rows,j -> column)

        # Rearrange b and c
        self.num_dt = sum(map(int, self.include_headnodes))
        if self.share_dt_for_two_graphs:
            assert self.share_BC_for_two_graphs == True
            self.num_dt = self.num_dt - 2

        dt = rearrange(
            wgt_params_data, 
            "b (i j) (n p q) -> p q b n i j", 
            i=self.image_height,
            j=self.image_width,
            n=self.num_heads,
            p=self.num_dt,
            q=2)
        
        if self.share_dt_for_two_graphs:
            if self.share_BC_for_two_graphs_mode == "line":
                dt = torch.repeat_interleave(dt, 2, dim=0)
            elif self.share_BC_for_two_graphs_mode == "diagonal":
                dt = torch.cat([dt, dt.flip(0)], dim=0)
            else:
                raise ValueError(f"Invalid share_BC_for_two_graphs_mode: {self.share_BC_for_two_graphs_mode}")

        dt = dt + self.dt_bias.unsqueeze(-1).unsqueeze(-1)
        dt = self.unused_dt_mask*F.softplus(dt)
        expdt = self.unused_dt_mask*torch.exp(-1*dt)
        
        if self.normalization_mode == "dt_self":
            dt_self = rearrange(
                dt_self, 
                "b (i j) (n p) -> p b n i j", 
                i=self.image_height,
                j=self.image_width,
                n=self.num_heads,
                p=1 if self.unified_view else sum(map(int, self.include_headnodes)),
            )
            dt_self = dt_self + self.dt_self_bias.unsqueeze(-1).unsqueeze(-1)
            dt_self = F.softplus(dt_self)
            
        # Rearrange b and c
        self.num_b_or_c = 1 if self.share_BC or self.unified_view else sum(map(int, self.include_headnodes))
        if self.share_BC_for_two_graphs:
           assert self.share_BC == False
           assert  sum(map(int, self.include_headnodes)) == 4
           self.num_b_or_c = self.num_b_or_c - 2

        b,c = bc
        b = b.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # shape: b n l d
        c = c.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # shape: b n l d

        b = rearrange(b, 'b n l (m d) -> m b n l d', m=self.num_b_or_c)
        c = rearrange(c, 'b n l (m d) -> m b n l d', m=self.num_b_or_c)

        if self.unified_view == False:
            # Get DAG mixers
            idx_dag_name = {0: 'top_left', 1: 'top_right', 2: 'bottom_left', 3: 'bottom_right'}
            bc_idx = 0
            dt_idx = 0 # serves as idx for both dt and dt_self
            output = torch.zeros_like(hidden_states_rearranged, device=device)
            I_matrix = torch.eye(self.image_height*self.image_width, device=device)

            for key in idx_dag_name.keys():
                if self.include_headnodes[key] == '1':
                    A_matrix, num_incident_edges = get_normalized_dag_mixer(expdt[dt_idx], idx_dag_name[key], device, True)
                    if self.use_fast_inverse:
                        matrix_mixer = DAGInverse.apply(A_matrix.to(torch.float32), self.image_height+self.image_width)
                    else:
                        matrix_mixer = torch.inverse(I_matrix - A_matrix)
                    
                    if self.normalization_mode == "dt_original":
                        summed_dt = torch.sum(dt[dt_idx], dim=0, keepdim=False) # (B, N, i, j)
                        norm_dt = summed_dt/torch.sqrt(num_incident_edges) #

                    elif self.normalization_mode == "sqrt":
                        summed_dt = torch.sum(dt[dt_idx], dim=0, keepdim=False)
                        root_dt = self.norm_sqrt_mul_factor*(1-torch.exp(2*summed_dt))
                        norm_dt = root_dt/torch.sqrt(num_incident_edges)
                    elif self.normalization_mode == "dt_self":
                        norm_dt = dt_self[dt_idx]/torch.sqrt(num_incident_edges)
                    else:
                        raise ValueError(f"Invalid normalization mode: {self.normalization_mode}")
                    
                    rearrange_norm_dt = rearrange(norm_dt, 'b n i j -> b n (i j)')
                    b_bar = torch.einsum('b n l d, b n l -> b n l d', b[bc_idx], rearrange_norm_dt)
                    output = output + torch.einsum('b n l d, b n l t, b n t d, b n t h -> b n l h',
                                                    c[bc_idx], matrix_mixer, b_bar, hidden_states_rearranged)
                    if not self.share_BC:
                        if self.share_BC_for_two_graphs:
                            if self.share_BC_for_two_graphs_mode == "line":
                                bc_idx += (int(key)%2 == 1)
                            elif self.share_BC_for_two_graphs_mode == "diagonal":
                                bc_idx = bc_idx if int(key)%2 == 1 else (bc_idx + 1)%2
                            else:
                                raise ValueError(
                                    f"Invalid share_BC_for_two_graphs_mode: {self.share_BC_for_two_graphs_mode}")
                        else:
                            bc_idx += 1

                    dt_idx += 1
           
            output = output/np.sqrt(num_matrix_mixers)
            
        else:
            assert self.share_BC == False
            idx_dag_name = {0: 'top_left', 1: 'top_right', 2: 'bottom_left', 3: 'bottom_right'}
            dt_idx = 0
            I_matrix = torch.eye(self.image_height*self.image_width, device=device)
            A_matrix = torch.zeros_like(I_matrix, device=device)

            for key in idx_dag_name.keys():
                if self.include_headnodes[key] == '1':
                    A_matrix_temp, num_incident_edges = get_normalized_dag_mixer(expdt[dt_idx], idx_dag_name[key], device, True)
                    A_matrix = A_matrix + A_matrix_temp
                    dt_idx += 1
            
            IA_matrix = I_matrix - A_matrix

            if self.normalization_mode == "dt_self":
                leq = rearrange(dt_self[0], 'b n i j -> b n (i j)')
                row_sum = torch.sum(IA_matrix, dim=-1, keepdim=False) + leq
                IA_matrix = self.inverse_factor*IA_matrix/row_sum.unsqueeze(-1)
            else:
                raise ValueError(f"Invalid normalization mode: {self.normalization_mode}")
            
            matrix_mixer = torch.inverse(IA_matrix)
            output = torch.einsum('b n l d, b n l t, b n t d, b n t h -> b n l h',
                                  c[0], matrix_mixer, b[0], hidden_states_rearranged)

        if self.debug_store_mm and str(device) == 'cuda:0' and self.epoch % 5000 == 1:
            # this is not mulitplied by dts: you have to do it manually
            # the constituent expdts are normalized
            np.save(f'matrix_mixer_{self.epoch}.npy', matrix_mixer.detach().cpu().numpy())
            # this is normalized using sqrt incident edges
            np.save(f'dts_{self.epoch}.npy', dt.detach().cpu().numpy())
            np.save(f'num_incident_{self.epoch}.npy', num_incident_edges.detach().cpu().numpy())

            # save B's and C's X's
            np.save(f'b_{self.epoch}.npy', b.to(torch.float32).detach().cpu().numpy())
            np.save(f'c_{self.epoch}.npy', c.to(torch.float32).detach().cpu().numpy())
            np.save(f'x_{self.epoch}.npy', hidden_states_rearranged.to(torch.float32).detach().cpu().numpy())

        output = self.std_dev*(output) + hidden_states_rearranged*self.D.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        final_output = rearrange(output, 'b n l h -> b l (n h)')

        return final_output
