import argparse

def add_mamba_args(parser):
    """Add Mamba-specific arguments for ablation studies.
    
    Ablation matrix:
        (default)              → transformer baseline
        --encoder-type mamba   → Mamba only
        --encoder-type text-gated-mamba    → Mamba + text gate
        --encoder-type multi-scale-mamba   → Mamba + text gate + multi-scale
    """
    parser.add_argument('--encoder-type', default='mamba',
                        choices=['transformer', 'mamba', 'bi-mamba', 'text-gated-mamba', 'multi-scale-mamba', 'hybrid', 'ava-hybrid'],
                        help='transformer: original baseline | mamba: SSM only | text-gated-mamba: +text gate | multi-scale-mamba: full model | hybrid: concat fusion | ava-hybrid: per-frame text-guided routing')
    parser.add_argument('--mamba-layers', default=2, type=int, help='Number of Mamba layers')
    parser.add_argument('--d-state', default=32, type=int, help='SSM state dimension (single-scale)')
    parser.add_argument('--d-state-list', default='8,16,32', type=str,
                        help='Comma-separated d_state values for multi-scale (short,mid,long)')
    parser.add_argument('--d-conv', default=4, type=int, help='Local convolution kernel width')
    parser.add_argument('--expand', default=2, type=int, help='Mamba block expansion factor')
    parser.add_argument('--gate-type', default='sigmoid', choices=['sigmoid', 'tanh', 'residual'],
                        help='Text gating activation function')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate on Mamba sub-layer outputs (default 0.0 = no dropout)')
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'],
                        help='Mamba architecture version (mamba1=S6, mamba2=SSD with larger state)')
    parser.add_argument('--mamba2-headdim', default=64, type=int,
                        help='Mamba2 head dimension (default 64)')
    parser.add_argument('--mamba2-ngroups', default=1, type=int,
                        help='Mamba2 number of SSM groups (default 1)')
    return parser

mamba_parser = add_mamba_args(argparse.ArgumentParser(description='Mamba options', add_help=False))
