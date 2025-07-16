import json
import os
from datetime import datetime

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore

from bert4rec import BERT4Rec
from config import DATA_CONFIG, MODEL_CONFIG, SYSTEM_CONFIG, TRAINING_CONFIG
from dataset import Bert4RecDataset
from train_utils import EarlyStopping, PrefetchLoader, TrainingHistory


class BERT4RecTrainer:
    def __init__(
        self,
        model: BERT4Rec,
        train_dataset: Bert4RecDataset,
        val_dataset: Bert4RecDataset,
        device: str = str(SYSTEM_CONFIG["device"]),
        num_epochs: int = TRAINING_CONFIG["num_epochs"],  # type: ignore
        batch_size: int = TRAINING_CONFIG["batch_size"],  # type: ignore
        learning_rate: float = TRAINING_CONFIG["learning_rate"],  # type: ignore
        adam_betas: tuple[float, float] = TRAINING_CONFIG["adam_betas"],  # type: ignore
        weight_decay: float = TRAINING_CONFIG["weight_decay"],  # type: ignore
        clip_grad_norm: float = TRAINING_CONFIG["clip_grad_norm"],  # type: ignore
        num_workers: int = TRAINING_CONFIG["num_workers"],  # type: ignore
    ):  # type: ignore
        self.model = model.to(device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.num_workers = num_workers

        # pin memory allocation for faster data transfer from CPU to GPU
        self.pin_memory = True if device == "cuda" else False

        # create a timestamped save dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(str(TRAINING_CONFIG["model_save_dir"]), timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Model will be saved to {self.save_dir}")

        # saving the training config
        config_path = os.path.join(self.save_dir, "config.json")
        config = {
            "model_config": MODEL_CONFIG,
            "data_config": DATA_CONFIG,
            "training_config": TRAINING_CONFIG,
            "system_config": SYSTEM_CONFIG,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )

        # linearly decrease the learning rate from 1.0 to 0.0 over the course of training
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.num_epochs,
        )

        self.history = TrainingHistory(self.save_dir)
        self.early_stopping = EarlyStopping()

    def get_dataloaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # number of subprocesses for data loading. More workers = faster data loading but more memory usage.
            pin_memory=self.pin_memory,
            drop_last=True,  # drops the last incomplete batch
            persistent_workers=True,  # keeps worker processes alive between epochs instead of recreating
            prefetch_factor=2,  # prefetch batches
        )

        val_loader = DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        if self.device == "cuda":
            self.train_loader = PrefetchLoader(train_loader)
            self.val_loader = PrefetchLoader(val_loader)

        return train_loader, val_loader

    def compute_metrics(self, src, tgt, src_mask):
        with torch.no_grad():
            logits = self.model.forward(
                src, src_mask
            )  # (batch_size, seq_len, vocab_size)

        mask_positions = src == self.model.mask_token_id  # (batch_size, seq_len)

        if not mask_positions.any():
            return {
                "hit_rate@1": 0.0,
                "hit_rate@5": 0.0,
                "hit_rate@10": 0.0,
                "num_masked_items": 0.0,
            }

        # extracting logits and targets for masked positions only
        masked_logits = logits[mask_positions]  # (num_masked_items, vocab_size)
        masked_targets = tgt[mask_positions]  # (num_masked_items,)

        # compute accuracy (hit rate @1)
        top_pred = torch.argmax(masked_logits, dim=-1)  # (num_masked_items,)
        accuracy = (top_pred == masked_targets).float().mean().item()

        # compute hit rate @5 and @10
        top_5_pred = torch.topk(masked_logits, k=5, dim=-1)[1]  # (num_masked_items, 5)
        top_10_pred = torch.topk(masked_logits, k=10, dim=-1)[
            1
        ]  # (num_masked_items, 10)
        # .unsqueeze(1) transforms masked_targets from (num_masked_items,) to (num_masked_items, 1), which allows for broadcasting
        hit_rate_5 = (
            (top_5_pred == masked_targets.unsqueeze(1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
        hit_rate_10 = (
            (top_10_pred == masked_targets.unsqueeze(1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

        return {
            "hit_rate@1": accuracy,
            "hit_rate@5": hit_rate_5,
            "hit_rate@10": hit_rate_10,
            "num_masked_items": len(masked_targets),
        }

    def validate(self, val_loader):
        total_val_loss = 0.0
        total_hit_rate_1 = 0.0
        total_hit_rate_5 = 0.0
        total_hit_rate_10 = 0.0
        total_num_masked_items = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in pbar:
                src = batch["source"].to(self.device)
                tgt = batch["target"].to(self.device)
                src_mask = batch["source_mask"].to(self.device)
                tgt_mask = batch["target_mask"].to(self.device)

                loss = self.model.compute_loss(
                    input_ids=src,
                    target_ids=tgt,
                    attention_mask=src_mask,
                    loss_mask=tgt_mask,
                )

                metrics = self.compute_metrics(src, tgt, src_mask)

                total_val_loss += loss.item()
                total_hit_rate_1 += metrics["hit_rate@1"]
                total_hit_rate_5 += metrics["hit_rate@5"]
                total_hit_rate_10 += metrics["hit_rate@10"]
                total_num_masked_items += metrics["num_masked_items"]
                
                # Update progress bar with current metrics
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'hr@1': f'{metrics["hit_rate@1"]:.3f}'
                })

        # compute average metrics
        avg_val_loss = total_val_loss / num_batches
        avg_hit_rate_1 = total_hit_rate_1 / num_batches
        avg_hit_rate_5 = total_hit_rate_5 / num_batches
        avg_hit_rate_10 = total_hit_rate_10 / num_batches
        avg_num_masked_items = total_num_masked_items / num_batches

        return {
            "val_loss": avg_val_loss,
            "hit_rate@1": avg_hit_rate_1,
            "hit_rate@5": avg_hit_rate_5,
            "hit_rate@10": avg_hit_rate_10,
            "num_masked_items": avg_num_masked_items,
        }

    def train(self):
        train_loader, val_loader = self.get_dataloaders()

        for epoch in range(self.num_epochs):
            total_train_loss = 0.0
            
            # Training loop with progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # move batch to device
                src = batch["source"].to(self.device)
                tgt = batch["target"].to(self.device)
                src_mask = batch["source_mask"].to(self.device)
                tgt_mask = batch["target_mask"].to(self.device)

                loss = self.model.compute_loss(
                    input_ids=src,
                    target_ids=tgt,
                    attention_mask=src_mask,
                    loss_mask=tgt_mask,
                )

                # zero out gradients from previous batch
                self.optimizer.zero_grad()
                # compute gradients through backprop
                loss.backward()
                # clip gradients to prevent exploding gradients
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                # backward() just calculates the gradients and step() actually does the update
                self.optimizer.step()

                total_train_loss += loss.item()
                
                # Update progress bar with current metrics
                current_lr = self.optimizer.param_groups[0]["lr"]
                avg_loss = total_train_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

            avg_train_loss = total_train_loss / len(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # validation
            self.model.eval()
            val_metrics = self.validate(val_loader)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  HR@1: {val_metrics['hit_rate@1']:.4f} | HR@5: {val_metrics['hit_rate@5']:.4f} | HR@10: {val_metrics['hit_rate@10']:.4f}")
            print(f"  LR: {current_lr:.2e} | Masked Items: {val_metrics['num_masked_items']:.1f}")

            # update training history
            self.history.update(epoch, avg_train_loss, val_metrics, current_lr)
            if self.early_stopping(val_metrics["val_loss"], self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # back to training mode
            self.model.train()
            # update learning rate
            self.scheduler.step()

        # save the model
        final_model_path = os.path.join(self.save_dir, "final_model.pth")
        
        # Use best weights if early stopping occurred, otherwise use current weights
        if self.early_stopping.early_stop and self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)
            print("Early stopping triggered - restored best model weights")
        
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Model saved to {final_model_path}")

        self.history.save_plots()
        self.history.save_history()

        return self.model


if __name__ == "__main__":
    import pandas as pd  # type: ignore

    from config import DATA_CONFIG
    from dataset import Bert4RecDataset

    dataset_name = "ml-1m"  # look for keys in DATASETS in config.py
    dataset_path = os.path.join(
        DATA_CONFIG["data_dir"], dataset_name, DATA_CONFIG["processed_ratings_file"]
    )  # type: ignore
    processed_ratings_df = pd.read_csv(dataset_path)

    dataset = Bert4RecDataset(processed_ratings_df, split_mode="train")
    val_dataset = Bert4RecDataset(processed_ratings_df, split_mode="valid")
    model = BERT4Rec()
    trainer = BERT4RecTrainer(model, dataset, val_dataset)
    trainer.train()
