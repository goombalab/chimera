from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_gt")
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = "SANLayer"

    # Number of Transformer layers in the model
    cfg.gt.layers = 4

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 4

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 96

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = False

    cfg.gt.residual = False

    # BigBird model/GPS-BigBird layer.
    cfg.gt.bigbird = CN()

    cfg.gt.bigbird.attention_type = "block_sparse"

    cfg.gt.bigbird.chunk_size_feed_forward = 0

    cfg.gt.bigbird.is_decoder = False

    cfg.gt.bigbird.add_cross_attention = False

    cfg.gt.bigbird.hidden_act = "relu"

    cfg.gt.bigbird.max_position_embeddings = 128

    cfg.gt.bigbird.use_bias = False

    cfg.gt.bigbird.num_random_blocks = 3

    cfg.gt.bigbird.block_size = 3

    cfg.gt.bigbird.layer_norm_eps = 1e-6

    cfg.gt.vn_pooling = "mean"

    # Decompose the original graph into DAG(s)
    cfg.gt.use_dag_decomposition = True

    # Number of DAGs to decompose the original graph into
    cfg.gt.num_dags = 1

    # Use Dense BFS which increases the number of edges in the DAG
    cfg.gt.if_dense_bfs = True

    # Number of parallel process for DAG Decomposition
    cfg.gt.num_workers = 0

    # Norm style and position
    cfg.gt.norm_style = "layer"  # "batch" or "layer"
    cfg.gt.local_norm_style = "layer"  # "batch" or "layer"
    cfg.gt.norm_position = "post"  # "pre" or "post"

    # Compose the local and global models
    cfg.gt.is_block_composed = False

    # Use FFN block in the GPS model
    cfg.gt.is_ffn_block = True

    # Number of attention heads in the Graph Transformer
    cfg.gt.d_inner = 96

    cfg.gt.ffn_d_dinner = 128

    cfg.gt.d_state = 80

    cfg.gt.gnn_layernorm = False

    cfg.gt.gnn_residual = False

    cfg.gt.add_conv_layers = False
    cfg.gt.conv_layers = 10

    cfg.gt.extra_normalizer = True
