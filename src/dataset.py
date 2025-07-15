import random
from typing import List

import pandas as pd
import torch
from config import MASK, MASKING_RATIO, MAX_SEQ_LEN, PAD
from torch.utils.data import Dataset


class Bert4RecDataset(Dataset):
    """
    This dataset class implements the data processing pipeline for training a BERT4Rec model,
    which adapts BERT's bidirectional self-attention mechanism for sequential recommendation tasks.

    Key Features:
    - Handles user-item interaction sequences
    - Implements dynamic masking for training
    - Supports different sequence padding modes
    - Provides train/validation/test split functionality
    - Filters interactions based on rating thresholds

    The main idea is to treat a user's interaction history as a sequence (like sentences in BERT),
    where items (movies, products, etc.) are tokens. The model learns to predict masked items
    based on the surrounding context.
    """

    def __init__(
        self,
        ratings_df,
        groupby_col="user_id",
        data_col="movie_mapped",
        target_column="rating",
        timestamp_col="timestamp",
        train_history: int = MAX_SEQ_LEN,
        valid_history: int = 5,
        padding_mode: str = "right",
        split_mode: str = "train",
        target_threshold=3,
        masking_ratio=MASKING_RATIO,
    ):
        """
        Initialize the BERT4Rec dataset.

        Args:
            ratings_df (pd.DataFrame): DataFrame containing user-item interactions
            groupby_col (str): Column name for user IDs
            data_col (str): Column name for item (movie in our case) IDs
            target_column (str): Column name for ratings/interaction values
            timestamp_col (str): Column name for interaction timestamps
            train_history (int): Maximum length of input sequences (200 as per the paper)
            valid_history (int): Number of items to reserve for validation/testing
            padding_mode (str): Whether to pad sequences from 'left' or 'right'
            split_mode (str): One of ['train', 'valid', 'test']
            target_threshold (float): Minimum rating threshold to consider an interaction positive
            masking_ratio (float): Probability of masking an item in the sequence (0.2 as per the paper)
        """
        super().__init__()

        self.ratings_df = ratings_df
        self.groupby_col = groupby_col
        self.data_col = data_col
        self.train_history = train_history
        self.valid_history = valid_history
        self.padding_mode = padding_mode
        self.split_mode = split_mode
        self.target_column = target_column
        self.target_threshold = target_threshold
        self.timestamp_col = timestamp_col

        # Filter interactions to focus on items the user actually liked
        # This is crucial as we want to predict items users will positively interact with
        if self.target_column:
            self.ratings_df = self.ratings_df[
                self.ratings_df[self.target_column] >= self.target_threshold
            ]
            self.ratings_df.reset_index(inplace=True)

        # Group interactions by user for efficient sequence extraction
        self.grouped_ratings = self.ratings_df.groupby(by=self.groupby_col)
        self.users = self.ratings_df[self.groupby_col].unique()
        self.masking_ratio = masking_ratio

    def pad_sequence(self, tokens: List, padding_mode: str = "right"):
        """
        Pad sequences to a fixed length for batch processing.

        Args:
            tokens (List): List of item IDs
            padding_mode (str): Whether to pad from 'left' or 'right'

        Returns:
            List: Padded sequence of length self.train_history

        Note:
            Right padding is more common, but left padding can help prevent the model
            from learning position-specific biases, especially for shorter sequences.
        """
        if len(tokens) < self.train_history:
            if padding_mode == "right":
                tokens = tokens + [PAD] * (self.train_history - len(tokens))
            else:
                tokens = [PAD] * (self.train_history - len(tokens)) + tokens
        return tokens

    def get_sequence(self, grouped_ratings: pd.DataFrame):
        """
        Extracts a user interaction sequence for BERT4Rec, simulating a session for training, validation, or testing.

        Intuition:
            - For "train": We want the model to generalize and not overfit to the very latest user interactions.
              So, we randomly select a recent subsequence (excluding the last `valid_history` items) to provide diverse training samples.
            - For "valid"/"test": We use the most recent interactions to evaluate the model's ability to predict future behavior.

        Args:
            grouped_ratings (pd.DataFrame): DataFrame containing all interactions for a single user, sorted by time.

        Raises:
            ValueError: If split_mode is not in ["train", "valid", "test"].

        Returns:
            pd.DataFrame: DataFrame of length up to `train_history`, containing the selected user sequence.
        """
        if self.split_mode == "train":
            # For training, we want to avoid using the very latest interactions (which are reserved for validation/testing).
            # So, we only sample up to (total interactions - valid_history).
            # This encourages the model to learn from various points in the user's history, not just the end.
            max_end = grouped_ratings.shape[0] - self.valid_history
            # Randomly select an end index for the training sequence, but ensure at least 10 interactions are included.
            end_ix = random.randint(10, max_end if max_end >= 10 else 10)
        elif self.split_mode in ["valid", "test"]:
            # For validation and test, we use the most recent interactions (the full available sequence).
            end_ix = grouped_ratings.shape[0]
        else:
            raise ValueError(
                f"Split should be either of `train`, `valid`, or `test`. {self.split_mode} is not supported"
            )

        # The start index is chosen so that the sequence is at most `train_history` long.
        # If there are fewer than `train_history` interactions, we start from the beginning (index 0).
        start_ix = max(0, end_ix - self.train_history)

        # Slice the DataFrame to get the sequence for this user.
        sequence = grouped_ratings[start_ix:end_ix]

        return sequence

    def mask_sequence(self, sequence: List, mask_ratio: float = 0.2):
        """
        Randomly mask items in a sequence for training.

        Args:
            sequence (List): List of item IDs
            mask_ratio (float): Probability of keeping an item unmasked

        Returns:
            List: Sequence with some items replaced by MASK token

        Note:
            This implements the masked item prediction task, similar to BERT's MLM.
            Items are randomly masked to force the model to use bidirectional context.
        """
        return [
            item if random.random() > self.masking_ratio else MASK for item in sequence
        ]

    def mask_sequence_last_items_only(self, sequence: List):
        """
        Mask the last N items in the sequence for validation/testing.

        Args:
            sequence (List): List of item IDs

        Returns:
            List: Sequence with last valid_history items partially masked

        Note:
            This is used for validation/testing to evaluate the model's ability
            to predict the user's most recent interactions.
        """
        return sequence[: -self.valid_history] + self.mask_sequence(
            sequence[-self.valid_history :], mask_ratio=0.5
        )

    def __getitem__(self, idx):
        """
        Get a training/validation/test sample.

        The process follows these steps:
        1. Get a user's interaction sequence
        2. Apply appropriate masking based on split_mode
        3. Pad sequences to fixed length
        4. Create attention masks for non-padded items
        5. Convert everything to PyTorch tensors

        Args:
            idx (int): Index of the user

        Returns:
            dict: Contains source and target sequences with their masks
                - source: Input sequence with masked items
                - target: Original sequence (ground truth)
                - source_mask: Attention mask for source sequence
                - target_mask: Attention mask for target sequence
        """
        # Get user's interaction sequence
        user = self.users[idx]
        grouped_rating = self.grouped_ratings.get_group(user)
        grouped_rating = grouped_rating.sort_values(by=self.timestamp_col).reset_index(
            drop=True
        )
        sequence = self.get_sequence(grouped_rating)
        target_items = sequence[self.data_col].tolist()

        # Apply masking strategy based on split mode
        if self.split_mode == "train":
            source_items = self.mask_sequence(target_items)
        else:
            source_items = self.mask_sequence_last_items_only(target_items)

        # Randomly choose padding mode to make model robust to both types
        pad_mode = "left" if random.random() < 0.5 else "right"
        target_items = self.pad_sequence(target_items, pad_mode)
        source_items = self.pad_sequence(source_items, pad_mode)

        # Create attention masks (1 for real items, 0 for padded items)
        # since both source and target are padded with PAD, both masks are the same
        target_mask = [1 if item != PAD else 0 for item in target_items]
        source_mask = [1 if item != PAD else 0 for item in source_items]

        # Convert everything to PyTorch tensors
        source_items = torch.tensor(source_items, dtype=torch.long)
        target_items = torch.tensor(target_items, dtype=torch.long)
        source_mask = torch.tensor(source_mask, dtype=torch.float)
        target_mask = torch.tensor(target_mask, dtype=torch.float)

        return {
            "source": source_items,
            "target": target_items,
            "source_mask": source_mask,
            "target_mask": target_mask,
        }

    def __len__(self):
        """Return the number of users in the dataset."""
        return len(self.users)
