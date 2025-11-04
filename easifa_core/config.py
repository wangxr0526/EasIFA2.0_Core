"""
Configuration for EasIFA inference.
"""
from pathlib import Path

# You should update these paths to where your checkpoints are located
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

# Default checkpoint paths - users should update these or pass custom paths
RXN_MODEL_PATH = PROJECT_ROOT_PATH / 'checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25'
ALL_FEATURES_EASIFA_MODEL_PATH = PROJECT_ROOT_PATH / 'checkpoints/active-site-categorie-prediction/train_in_uniprot_ecreact_cluster_split_merge_dataset_all_limit_100_at_2025-04-05-19-32-45/global_step_210000'
WO_STRUCTURES_EASIFA_MODEL_PATH = PROJECT_ROOT_PATH / 'checkpoints/active-site-categorie-prediction-sequence/train_in_uniprot_ecreact_cluster_split_merge_dataset_all_limit_100_at_2025-04-08-11-09-30/global_step_170000'
WO_RXN_EASIFA_MODEL_PATH = PROJECT_ROOT_PATH / 'checkpoints/active-site-categorie-prediction-norxn/train_in_uniprot_ecreact_cluster_split_merge_dataset_all_limit_100_at_2025-04-09-10-28-44/global_step_100000'
WO_RXN_STRUCTURE_EASIFA_MODEL_PATH = PROJECT_ROOT_PATH / 'checkpoints/active-site-categorie-prediction-norxn-sequence/train_in_uniprot_ecreact_cluster_split_merge_dataset_all_limit_100_at_2025-04-10-16-15-23/global_step_100000'


class EasIFAInferenceConfig:
    """
    Configuration class for EasIFA inference.
    
    Attributes:
        model_to_use: List of models to load and use for inference
        use_gpu: Whether to use GPU for inference
        max_enzyme_aa_length: Maximum length of enzyme amino acid sequence
        bridge_hidden_dim: Hidden dimension for bridge layers
        num_active_site_type: Number of active site types (4: non-site, BINDING, ACT_SITE, SITE)
        model_state_paths: Dictionary mapping model names to checkpoint paths
        gpu_allocations: Dictionary mapping model names to GPU device IDs or 'cpu'
        pred_tolist: Whether to convert predictions to list format
    """
    
    model_to_use = [
        "all_features",        # Full model with structure + reaction
        "wo_structures",       # Sequence + reaction
        "wo_reactions",        # Structure only
        "wo_rxn_structures",   # Sequence only
    ]
    
    use_gpu = False  # Set to True to use GPU by default
    max_enzyme_aa_length = 1000
    bridge_hidden_dim = 128
    num_active_site_type = 4
    
    model_state_paths = {
        "all_features": ALL_FEATURES_EASIFA_MODEL_PATH,
        "wo_structures": WO_STRUCTURES_EASIFA_MODEL_PATH,
        "wo_reactions": WO_RXN_EASIFA_MODEL_PATH,
        "wo_rxn_structures": WO_RXN_STRUCTURE_EASIFA_MODEL_PATH,
        "rxn_model_path": RXN_MODEL_PATH,
    }
    
    gpu_allocations = {
        "all_features": 'cpu',         # cpu/-1 or gpu_id (e.g., 0, 1, 2, ...)
        "wo_structures": 'cpu',        # cpu/-1 or gpu_id
        "wo_reactions": 'cpu',         # cpu/-1 or gpu_id
        "wo_rxn_structures": 'cpu',    # cpu/-1 or gpu_id
    }
    
    pred_tolist = True

    def __init__(self, **kwargs):
        """
        Initialize config with optional keyword arguments to override defaults.
        
        Args:
            **kwargs: Any config attribute to override (e.g., use_gpu=True, max_enzyme_aa_length=500)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"EasIFAInferenceConfig has no attribute '{key}'")
