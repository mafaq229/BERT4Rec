import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from typing import Optional

from dataset import Bert4RecDataset
from bert4rec import BERT4Rec, create_bert4rec_model


def train_bert4rec(
    model: BERT4Rec,
    train_dataset: Bert4RecDataset,
    valid_dataset: Optional[Bert4RecDataset] = None,
    batch_size: int = 256,
    lr: float = 1e-4,
    epochs: int = 10,
    device: str = "cpu",
    num_workers: int = 4,
    max_grad_norm: float = 1.0,
    scheduler=None
) -> BERT4Rec:
    """
    Train a BERT4Rec model on the given datasets.

    Args:
        model: BERT4Rec instance
        train_dataset: Dataset split with split_mode='train'
        valid_dataset: Dataset split with split_mode='valid' or None
        batch_size: Batch size for DataLoader
        lr: Learning rate
        epochs: Number of training epochs
        device: Device string (e.g., 'cpu', 'cuda', or 'mps')
        num_workers: DataLoader num_workers
        max_grad_norm: Maximum gradient norm for clipping
        scheduler: Optional LR scheduler (e.g., torch.optim.lr_scheduler)

    Returns:
        Trained BERT4Rec model
    """
    # Prepare DataLoaders
    pin_memory = device != "mps"  # Disable pin_memory for MPS (Apple Silicon)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # Move model to device
    model.to(device)
    model.train()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_train_loss = 0.0

        for batch in train_loader:
            # Move batch to device
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)
            src_mask = batch["source_mask"].to(device)
            tgt_mask = batch["target_mask"].to(device)

            # Compute loss
            loss = model.compute_loss(
                input_ids=src,
                target_ids=tgt,
                attention_mask=src_mask,
                loss_mask=tgt_mask
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        if valid_dataset is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    src = batch["source"].to(device)
                    tgt = batch["target"].to(device)
                    src_mask = batch["source_mask"].to(device)
                    tgt_mask = batch["target_mask"].to(device)

                    loss = model.compute_loss(
                        input_ids=src,
                        target_ids=tgt,
                        attention_mask=src_mask,
                        loss_mask=tgt_mask
                    )
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(valid_loader)
            print(f"Epoch {epoch}/{epochs} | Val   Loss: {avg_val_loss:.4f}")
            model.train()

        # Step LR scheduler if provided
        if scheduler is not None:
            scheduler.step()

    return model


if __name__ == "__main__":
    import pandas as pd
    from config import MODEL_CONFIG, DATA_CONFIG, SYSTEM_CONFIG
    import os

    # Extract constants from config
    VOCAB_SIZE = MODEL_CONFIG['vocab_size']
    MAX_SEQ_LEN = MODEL_CONFIG['max_seq_len']
    D_MODEL = MODEL_CONFIG['d_model']
    N_HEADS = MODEL_CONFIG['n_heads']
    N_LAYERS = MODEL_CONFIG['n_layers']
    D_FF = MODEL_CONFIG['d_ff']
    DROPOUT = MODEL_CONFIG['dropout']
    PAD_TOKEN_ID = MODEL_CONFIG['pad_token_id']
    MASK_TOKEN_ID = MODEL_CONFIG['mask_token_id']
    DEVICE = SYSTEM_CONFIG['device']
    
    # Construct data path
    DATA_PATH = "/Users/muhammadafaq/Documents/research/gatech/Deep Learning/project/artifacts/bert4rec/data/ratings_mapped.csv"

    # Load your interaction data
    df = pd.read_csv(DATA_PATH)

    # Create datasets using the same dataframe but different split modes
    # This uses temporal splitting: train on older interactions, validate on recent ones
    train_ds = Bert4RecDataset(
        df,
        split_mode='train'
    )
    valid_ds = Bert4RecDataset(
        df,
        split_mode='valid'
    )

    # Initialize model
    model = create_bert4rec_model(
        item_vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pad_token_id=PAD_TOKEN_ID,
        mask_token_id=MASK_TOKEN_ID
    )

    # Train
    trained_model = train_bert4rec(
        model,
        train_ds,
        valid_dataset=valid_ds,
        batch_size=128,
        lr=1e-4,
        epochs=10,
        device=DEVICE
    )

    # Save checkpoint
    torch.save(trained_model.state_dict(), "bert4rec_trained.pth")
