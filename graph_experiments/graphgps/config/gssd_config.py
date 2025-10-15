from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_gssd")
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gssd = CN()

    cfg.gssd.d_state = 80

    cfg.gssd.d_inner = 96

    cfg.gssd.d_conv = 4

    cfg.gssd.headdim = 48

    cfg.gssd.layer_args = CN()

    cfg.gssd.layer_args.conv_type = "identity"

    cfg.gssd.layer_args.broadcast_bc_in_heads = True

    cfg.gssd.layer_args.norm_constraint = "sum_leq"

    cfg.gssd.layer_args.sub_constraint = "max"

    cfg.gssd.layer_args.combine_type = "sum_node_edge"

    cfg.gssd.layer_args.use_fast_inverse = False

    cfg.gssd.layer_args.approx_factor = 1.0

    cfg.gssd.layer_args.dt_multiplier = 1

    cfg.gssd.layer_args.normalize_L = False

    cfg.gssd.layer_args.normalize_x = False

    cfg.gssd.layer_args.dt_max = 0.1

    cfg.gssd.layer_args.dt_min = 0.005

    cfg.name_tag = ""

    cfg.gssd.layer_args.is_share_BC_dags = True  # share BC across DAGs

    cfg.gssd.layer_args.is_edge_dt = False  # use edge DT

    # DAG Flags - Can remove

    cfg.gssd.layer_args.is_share_BC_heads = True  # share BC across heads

    cfg.gssd.layer_args.is_sum_leq = True  # use node DT

    # Unused Flags

    cfg.gssd.layer_args.normalize_L = False

    cfg.gssd.layer_args.normalize_x = False

    cfg.gssd.layer_args.graph_diameter = 100

    cfg.gssd.layer_args.norm_style = "prenorm"
