import json
import os

import matplotlib.pyplot as plt
import torch
from torch.cuda import Stream

from config import TRAINING_CONFIG


class PrefetchLoader:
    """DataLoader wrapper that prefetches data to GPU"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = Stream()
        
    def __iter__(self):
        """
        Iterator that prefetches batches to GPU asynchronously.
        Uses CUDA streams to overlap data transfer with computation.
        Each batch is moved to GPU in non-blocking mode for better performance.
        """
        for batch in self.dataloader:
            with torch.cuda.stream(self.stream): # type: ignore
                batch = [item.cuda(non_blocking=True) for item in batch]
            yield batch
            
    def __len__(self):
        return len(self.dataloader)


class TrainingHistory:
    """Track and visualize training history"""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "hit_rate@1": [],
            "hit_rate@5": [],
            "hit_rate@10": [],
            "num_masked_items": [],
            "epochs": [],
            "learning_rates": []
        } # type: ignore
        os.makedirs(self.save_dir, exist_ok=True)
        
    def update(self, epoch, train_loss, val_metrics, current_lr):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_metrics["val_loss"])
        self.history["hit_rate@1"].append(val_metrics["hit_rate@1"])
        self.history["hit_rate@5"].append(val_metrics["hit_rate@5"])
        self.history["hit_rate@10"].append(val_metrics["hit_rate@10"])
        self.history["num_masked_items"].append(val_metrics["num_masked_items"])
        self.history["epochs"].append(epoch)
        self.history["learning_rates"].append(current_lr)
        
    def save_plots(self):
        """Save training curves as plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.history['epochs'], self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(self.history['epochs'], self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Hit Rate@1/Accuracy curve
        axes[0, 1].plot(self.history['epochs'], self.history['hit_rate@1'], 'g-', label='Hit Rate@1/Accuracy')
        axes[0, 1].set_title('Hit Rate@1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Hit Rate@1')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Hit Rate@5 and Hit Rate@10 curves
        axes[1, 0].plot(self.history['epochs'], self.history['hit_rate@5'], 'm-', label='Hit Rate@5')
        axes[1, 0].plot(self.history['epochs'], self.history['hit_rate@10'], 'c-', label='Hit Rate@10')
        axes[1, 0].set_title('Hit Rate@5 and Hit Rate@10')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Hit Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve

        axes[1, 1].plot(self.history['epochs'], self.history['learning_rates'], 'orange', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')

        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    📊 Training curves saved to {os.path.join(self.save_dir, 'training_curves.png')}")
    
    def save_history(self):
        """Save training history as JSON."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"    💾 Training history saved to {history_path}")
        

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, 
                 patience: int = TRAINING_CONFIG['patience'],  # type: ignore
                 min_delta: float = TRAINING_CONFIG['min_delta'],  # type: ignore
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, score, model) -> bool:
        """Score is the validation loss so lower is better"""
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
        
    def _save_checkpoint(self, model):
        """Save best model weights"""
        if self.restore_best_weights:
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            