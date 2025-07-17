"""
Configuration file for BERT4Rec model training on MovieLens 1M dataset.
Parameters optimized based on the original BERT4Rec paper.
"""

import torch
import os

DATASETS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}
DATA_SAVE_PATH = "artifacts/data"


# =============================================================================
# SPECIAL TOKENS
# =============================================================================
PAD = 0     # Padding token for sequences
MASK = 1    # Mask token for masked language modeling
START = 2   # Start token for movie indexing


# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_CONFIG = {
    # File paths
    'data_dir': DATA_SAVE_PATH,
    'processed_ratings_file': 'ratings_mapped.csv',
    'processed_movies_file': 'movies_mapped.csv',
    'processed_users_file': 'users.csv',
    'new_to_old_movie_id_mapping_file': 'new_to_old_movie_id_mapping.pkl',

    # Columns
    'user_col': 'user_id',
    'item_col': 'movie_mapped',
    'rating_col': 'rating',
    'timestamp_col': 'timestamp',
    
    # Data processing parameters
    # do some stats on the sequence lengths and see if max_seq_len is too high i.e. more than 50 percentile
    'max_seq_len': 200, # as per BERT4Rec paper. 
    'valid_history': 5,  # count of last interactions reserved and excluded from training sequences
    'masking_ratio': 0.2, # as per BERT4Rec paper. 
    'positive_review_threshold': False,
    'target_threshold': 3,
    'min_real_interactions': 10, # minimum number of real interactions in a sequence
}

# =============================================================================
# MODEL CONFIGURATION - Optimized for MovieLens 1M based on BERT4Rec paper
# =============================================================================
MODEL_CONFIG = {
    # Core architecture parameters (from BERT4Rec paper)
    'd_model': 64,           # Hidden dimension
    'n_heads': 2,            # Number of attention heads
    'n_layers': 2,           # Number of transformer layers
    'd_ff': 256,             # Feed-forward dimension
    'dropout': 0.2,          # Dropout rate
    
    # Sequence parameters
    'max_seq_len': DATA_CONFIG['max_seq_len'],      # Maximum sequence length (as per paper)
    'masking_ratio': DATA_CONFIG['masking_ratio'],    # Probability of masking items (as per paper)
    
    # Vocabulary parameters (should be updated based on dataset used)
    'vocab_size': 3885,      # ML-1M has 3884 (movies + special tokens). 3416 in paper
    'pad_token_id': PAD,
    'mask_token_id': MASK,
}

# =============================================================================
# TRAINING CONFIGURATION - Optimized for MovieLens 1M
# =============================================================================
TRAINING_CONFIG = {
    # Optimization parameters (from BERT4Rec paper)
    'learning_rate': 0.0001,
    'adam_betas': (0.9, 0.98),
    'batch_size': 256,
    'num_epochs': 2000,
    # 'warmup_steps': 100,
    'weight_decay': 0.01,
    'clip_grad_norm': 5.0, # as per BERT4Rec paper (can try 1.0 as well)
    
    # Linear learning rate scheduling (as per BERT4Rec paper)
    'min_lr': 1e-6,              # Minimum learning rate for linear scheduler
    
    # Early stopping
    'patience': 50,              # Early stopping patience
    'min_delta': 0.001,          # Minimum improvement for early stopping
    
    # Validation parameters
    'val_every_n_epochs': 1,     # Validate every N epochs
    'save_every_n_epochs': 5,    # Save checkpoint every N epochs
    
    # Data parameters
    'valid_history': 5,          # Number of items for validation
    'target_threshold': 3,       # Minimum rating threshold (ML-1M uses 1-5 scale)
    'num_workers': 4,            # DataLoader workers
    
    # Loss computation
    'label_smoothing': 0.1,      # Label smoothing for cross-entropy loss
    
    # save_path
    'model_save_dir': 'artifacts/model'
}

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
SYSTEM_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'seed': 42,
    'fp16': torch.cuda.is_available(),  # Only use mixed precision on CUDA
    'gradient_clip_norm': 1.0,   # Gradient clipping norm
    'log_level': 'INFO',
    'save_path': 'artifacts/bert4rec',
    'experiment_name': 'bert4rec_ml1m',
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
EVAL_CONFIG = {
    'top_k': [1, 5, 10, 20],    # Top-K for evaluation metrics
    'metrics': ['hit_rate', 'ndcg', 'mrr'],  # Evaluation metrics
    'eval_batch_size': 512,      # Batch size for evaluation
}

# =============================================================================
# COMBINED CONFIGURATION
# =============================================================================
def get_ml1m_config():
    """Get complete configuration for MovieLens 1M dataset."""
    return {
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'data': DATA_CONFIG,
        'system': SYSTEM_CONFIG,
        'eval': EVAL_CONFIG,
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def create_save_directory(save_path: str):
    """Create save directory if it doesn't exist."""
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'logs'), exist_ok=True)

def update_vocab_size(vocab_size: int):
    """Update vocabulary size based on actual dataset."""
    MODEL_CONFIG['vocab_size'] = vocab_size

# Legacy constants for backward compatibility
MAX_SEQ_LEN = MODEL_CONFIG['max_seq_len']
MASKING_RATIO = MODEL_CONFIG['masking_ratio']
save_path = SYSTEM_CONFIG['save_path']

