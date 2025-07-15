"""
Implementation based on "BERT4Rec: Sequential Recommendation with Bidirectional 
Encoder Representations from Transformer" and "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Dict, Any, List


# =============================================================================
# PART 1: CORE UTILITIES
# =============================================================================
# These are fundamental building blocks used throughout the transformer architecture


def clones(module, N):
    """
    Create N identical copies of a PyTorch module.
    
    Purpose: In transformers, we need multiple identical layers (e.g., N encoder layers).
    This function creates deep copies to ensure each layer has independent parameters.
    
    Intuition: Think of this as creating N photocopies of a blueprint - each copy
    can be modified independently without affecting others.
    
    Args:
        module: The PyTorch module to clone
        N: Number of copies to create
        
    Returns:
        nn.ModuleList: List of N independent copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    Layer Normalization: Normalizes inputs across the feature dimension.
    
    Purpose: Stabilizes training by normalizing inputs to have zero mean and unit variance.
    Unlike batch normalization (which normalizes across the batch), layer normalization
    normalizes across the features for each individual sample.
    
    Intuition: Imagine you have test scores for different subjects. Layer normalization
    would standardize each student's scores across all subjects, making them comparable.
    
    Mathematical Formula: 
        output = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    Why it helps:
    - Prevents internal covariate shift
    - Allows higher learning rates
    - Reduces sensitivity to initialization
    """
    
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        # Learnable parameters for scaling (gamma) and shifting (beta)
        self.a_2 = nn.Parameter(torch.ones(features))   # gamma (scale)
        self.b_2 = nn.Parameter(torch.zeros(features))  # beta (shift)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate mean and standard deviation across the last dimension (features)
        mean = x.mean(-1, keepdim=True)  # Shape: (batch, seq_len, 1)
        std = x.std(-1, keepdim=True)    # Shape: (batch, seq_len, 1)
        
        # Apply normalization with learnable parameters
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual Connection with Layer Normalization and Dropout.
    
    Purpose: This is the "Add & Norm" component that wraps each sublayer (attention, FFN)
    in the transformer. It applies layer normalization, then the sublayer, then dropout,
    and finally adds the residual connection.
    
    Intuition: Think of this as a "safety net" for information flow. The residual connection
    ensures that even if a layer learns nothing useful, the original information can still
    pass through. It's like having a highway with both local roads and express lanes.
    
    Why residual connections matter:
    - Solve vanishing gradient problem in deep networks
    - Allow training of very deep networks (50+ layers)
    - Provide direct paths for gradient flow
    
    Formula: x + dropout(sublayer(norm(x)))
    """
    
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        
        Args:
            x: Input tensor
            sublayer: Function that processes the input (e.g., attention, FFN)
        """
        # even though the Bert4Rec paper uses the second version, the first version performs better
        return x + self.dropout(sublayer(self.norm(x))) # (based on harvard annotated transformer)
        # return self.norm(x + self.dropout(sublayer(x)))  # based on original paper


# =============================================================================
# PART 2: EMBEDDINGS AND POSITIONAL ENCODING
# =============================================================================
# These components convert discrete tokens to continuous representations and add position info


class Embeddings(nn.Module):
    """
    Token Embeddings: Convert discrete item IDs to continuous vector representations.
    
    Purpose: Neural networks work with continuous values, but items are discrete IDs.
    Embeddings learn a mapping from item IDs to dense vectors that capture semantic
    relationships between items.
    
    Intuition: Like a dictionary that maps words to their meanings, embeddings map
    item IDs to vectors that represent their "meaning" in the recommendation context.
    Similar items (e.g., action movies) will have similar embedding vectors.
    
    Why scaling by sqrt(d_model):
    - Prevents embeddings from being too large when d_model is large
    - Maintains proper balance with positional encodings
    - Follows the original transformer paper
    """
    
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)  # Lookup table
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings and apply scaling.
        
        Args:
            x: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding: Use trainable embedding vectors for positions.
    
    Purpose: Instead of fixed sinusoidal patterns, learn position representations
    that are optimized specifically for the recommendation task. This allows the
    model to adapt position encoding to the specific patterns in user behavior.
    
    Intuition: Like learning a personalized calendar system where each position
    has a unique learned "signature" that captures its importance in user sequences.
    The model learns which positions are most informative for recommendations.
    
    Why learned embeddings:
    - Adapt to specific patterns in sequential recommendation data
    - Can capture domain-specific positional relationships
    - More flexible than fixed mathematical functions
    - Used in original BERT and BERT4Rec papers
    
    Architecture:
    - Embedding layer: position_id -> position_embedding
    - Same dimension as token embeddings for direct addition
    - Trainable parameters optimized during training
    """
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Create learnable position embedding table
        # Each position gets its own learned vector
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with learned positional information added
        """
        batch_size, seq_len, d_model = x.size()
        
        # (seq_len) -> (1, seq_len) -> (batch_size, seq_len)
        # batch_size argument is the number of rows to expand to. -1 in expand means to not change this dimension. 
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        position_embeddings = self.position_embeddings(position_ids)
        
        x = x + position_embeddings
        
        return self.dropout(x)


# =============================================================================
# PART 3: ATTENTION MECHANISM
# =============================================================================
# The core innovation of transformers: allowing each position to attend to all others


def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention: The core attention mechanism.
    
    Purpose: Allow each position in the sequence to attend to (look at) all other positions.
    This creates rich representations where each item's representation is influenced by
    all other items in the sequence.
    
    Intuition: Like a group discussion where each person can listen to everyone else
    and decide how much to focus on each speaker based on relevance. The "queries" are
    questions, "keys" are topics people can talk about, and "values" are what they actually say.
    
    Why "scaled" dot-product:
    - Dot product measures similarity between query and key
    - Scaling by sqrt(d_k) prevents values from becoming too large
    - Large values can push softmax into saturation regions with small gradients
    
    Mathematical steps:
    1. Compute attention scores: QK^T / sqrt(d_k)
    2. Apply mask (if provided) to prevent attending to certain positions
    3. Apply softmax to get attention weights (probabilities)
    4. Multiply by values to get weighted output
    
    Args:
        query: What we're looking for (batch_size, h, seq_len, d_k)
        key: What we're looking in (batch_size, h, seq_len, d_k)
        value: What we actually get (batch_size, h, seq_len, d_k)
        mask: Which positions to ignore (batch_size, 1, 1, seq_len) - 1 for valid, 0 for padding
        dropout: Regularization for attention weights
    
    Returns:
        (output, attention_weights): The attended representation and attention scores
    """
    d_k = query.size(-1)
    
    # Step 1: Compute attention scores (similarity between queries and keys)
    # Shape: (batch_size, h, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Step 2: Apply mask (set masked positions to very negative values)
    if mask is not None:
        # Expand mask to match attention scores shape
        # mask: (batch_size, 1, 1, seq_len) -> (batch_size, h, seq_len, seq_len)
        try:
            expanded_mask = mask.expand(-1, scores.size(1), scores.size(2), -1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        except RuntimeError as e:
            # Handle dimension mismatch - create a basic key padding mask
            if mask.dim() == 2:  # (batch_size, seq_len)
                # Convert to (batch_size, 1, 1, seq_len) and expand
                mask = mask.unsqueeze(1).unsqueeze(1)
                expanded_mask = mask.expand(-1, scores.size(1), scores.size(2), -1)
                scores = scores.masked_fill(expanded_mask == 0, -1e9)
            else:
                raise e
    
    # Step 3: Apply softmax to get attention probabilities
    p_attention = scores.softmax(dim=-1)
    
    # Step 4: Apply dropout for regularization
    if dropout is not None:
        p_attention = dropout(p_attention)
    
    # Step 5: Multiply by values to get final output
    return torch.matmul(p_attention, value), p_attention


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention: Run multiple attention heads in parallel.
    
    Purpose: Different attention heads can focus on different types of relationships:
    - Head 1 might focus on sequential patterns (item A often follows item B)
    - Head 2 might focus on categorical similarity (both are action movies)
    - Head 3 might focus on temporal patterns (both watched on weekends)
    
    Intuition: Like having multiple experts examining the same data from different
    perspectives. Each expert (head) specializes in finding different types of patterns.
    
    Why multiple heads:
    - Allows the model to jointly attend to different representation subspaces
    - Each head can learn different types of relationships
    - Provides richer representations than single-head attention
    
    Architecture:
    1. Project Q, K, V to multiple heads using linear layers
    2. Apply scaled dot-product attention to each head independently
    3. Concatenate all head outputs
    4. Apply final linear projection
    
    Formula: MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h  # Dimension per head
        self.h = h
        
        # Four linear layers: Q, K, V projections + output projection
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            query, key, value: Input tensors (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        if mask is not None:
            # Convert padding mask to attention mask
            # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        nbatches = query.size(0)
        
        # Step 1: Apply linear projections and reshape for multi-head
        # From (batch_size, seq_len, d_model) to (batch_size, h, seq_len, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        # Step 2: Apply attention to each head
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Step 3: Concatenate heads and apply final linear layer
        # From (batch_size, h, seq_len, d_k) to (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)


# =============================================================================
# PART 4: FEED-FORWARD NETWORK
# =============================================================================
# Position-wise processing to add non-linearity and increase model capacity


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: Add non-linearity and model capacity.
    
    Purpose: While attention is great for modeling relationships between positions,
    it's essentially linear. The FFN adds non-linearity and increases the model's
    capacity to learn complex patterns.
    
    Intuition: After gathering information from all positions via attention, each
    position needs to "think" about what it learned. The FFN is like a small neural
    network that processes each position independently.
    
    Why position-wise:
    - Applied to each position separately and identically
    - Same parameters used for all positions
    - Allows parallel processing across positions
    
    Architecture:
    - Linear layer: d_model -> d_ff (expansion)
    - ReLU activation (non-linearity)
    - Dropout (regularization)
    - Linear layer: d_ff -> d_model (compression)
    
    Formula: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# =============================================================================
# PART 5: ENCODER LAYER AND ENCODER
# =============================================================================
# Combining attention and feed-forward into encoder layers, then stacking them


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer: Self-attention + Feed-forward with residual connections.
    
    Purpose: One layer of the encoder that combines self-attention and feed-forward
    processing. Each layer allows the model to build increasingly complex representations.
    
    Intuition: Like a single step in a complex reasoning process:
    1. First, look at all items and understand their relationships (self-attention)
    2. Then, process this information to extract insights (feed-forward)
    3. Keep the original information to prevent forgetting (residual connections)
    
    Why this order:
    - Self-attention first: Gather information from all positions
    - Feed-forward second: Process the gathered information
    - Residual connections throughout: Ensure information flow
    
    Architecture:
    Input -> LayerNorm -> Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm -> Output
    """
    
    def __init__(self, d_model: int, self_attention: MultiHeadedAttention, 
                 feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply encoder layer: self-attention followed by feed-forward.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention: query, key, and value are all the same (x)
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        # Feed-forward processing
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Stack of N Encoder Layers: Build deep representations through multiple layers.
    
    Purpose: Stack multiple encoder layers to build increasingly complex and abstract
    representations. Each layer can capture different levels of patterns:
    - Early layers: Simple patterns (item A follows item B)
    - Middle layers: Complex patterns (genre preferences, seasonal trends)
    - Late layers: Abstract patterns (user behavior archetypes)
    
    Intuition: Like a deep conversation where each exchange adds more nuance and
    understanding. The first layer might identify basic patterns, while deeper
    layers discover subtle relationships and complex user preferences.
    
    Why multiple layers:
    - Allows hierarchical feature learning
    - Each layer can build upon the previous layer's representations
    - Enables the model to capture both local and global patterns
    
    Architecture:
    Input -> [EncoderLayer -> EncoderLayer -> ... -> EncoderLayer] -> LayerNorm -> Output
    """
    
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pass input through N encoder layers followed by normalization.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# PART 6: BERT4REC MODEL
# =============================================================================
# The complete BERT4Rec model combining all components


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation
    
    Purpose: Learn user preferences by predicting masked items in interaction sequences.
    Unlike traditional sequential models that only look at past items, BERT4Rec uses
    bidirectional attention to consider both past and future context.
    
    Key Innovation: Masked Item Prediction
    - Randomly mask some items in user sequences during training
    - Predict masked items based on surrounding context
    - This teaches the model to understand item relationships and user preferences
    
    Why Bidirectional:
    - Traditional models: item₁ → item₂ → item₃ → ? (only past context)
    - BERT4Rec: item₁ ← item₂ ← item₃ → item₄ → item₅ (full context)
    - Richer understanding of user preferences and item relationships
    
    Architecture Overview:
    Input Sequence: [item₁, item₂, [MASK], item₄, item₅]
    ↓
    Item Embeddings + Positional Encoding
    ↓
    N × Encoder Layers (Self-Attention + Feed-Forward)
    ↓
    Output Head (Linear layer to item vocabulary)
    ↓
    Predictions: [item₁, item₂, item₃, item₄, item₅]
    
    Training Process:
    1. Take user interaction sequence: [movie_A, movie_B, movie_C, movie_D]
    2. Randomly mask items: [movie_A, [MASK], movie_C, movie_D]
    3. Predict masked items using bidirectional context
    4. Learn from prediction errors to improve recommendations
    
    Benefits:
    - Captures complex item relationships
    - Handles sparse data better than collaborative filtering
    - Provides interpretable attention weights
    - Scales to large vocabularies
    """
    
    def __init__(
        self,
        item_vocab_size: int = 3885, # 3884 is the max mapped movie id which should be the minimum size of the embedding layer
        max_seq_len: int = 200,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        pad_token_id: int = 0,
        mask_token_id: int = 1
    ):
        """
        Initialize BERT4Rec model.
        
        Args:
            item_vocab_size: Number of unique items (+ special tokens)
            max_seq_len: Maximum sequence length
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward network dimension (intermediate layer size)
            dropout: Dropout probability
            pad_token_id: ID for padding token
            mask_token_id: ID for mask token
        """
        super().__init__()
        
        # Store configuration
        self.item_vocab_size = item_vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # Create shared components
        c = copy.deepcopy
        attention = MultiHeadedAttention(self.n_heads, self.d_model, self.dropout)
        feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position_encoding = LearnedPositionalEncoding(self.max_seq_len, self.d_model, self.dropout)
        
        # Build model components
        # The embedding layer maps input item IDs (shape: [batch_size, seq_len]) to dense vectors
        # of shape [batch_size, seq_len, d_model]. The positional encoding then adds position
        # information to each embedding, keeping the shape unchanged. This nn.Sequential
        # ensures that input token IDs are first converted to embeddings, then positionally
        # encoded, resulting in a tensor of shape [batch_size, seq_len, d_model] ready for the encoder.
        self.item_embeddings = Embeddings(self.d_model, self.item_vocab_size)
        self.embeddings = nn.Sequential(
            self.item_embeddings,  # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
            position_encoding   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        )
        
        encoder_layer = EncoderLayer(self.d_model, c(attention), c(feed_forward), self.dropout)
        self.encoder = Encoder(encoder_layer, self.n_layers)
        
        # The final layer projects the model output back to d_model dimensions (not to item_vocab_size).
        # This is used as an intermediate transformation before the output projection.
        self.final_layer = nn.Linear(self.d_model, self.d_model)  # W_P and b_P
        self.output_bias = nn.Parameter(torch.zeros(self.item_vocab_size))  # b_O
        
        # initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with truncated normal distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                # in paper, all parameters are initialized with truncated normal distribution
                nn.init.trunc_normal_(p, std=0.02)
                # nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask to ignore padding tokens.
        
        Purpose: Prevent the model from attending to padding tokens, which are
        meaningless filler tokens used to make sequences the same length.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            
        Returns:
            Attention mask (batch_size, seq_len) where 1 = valid, 0 = padding
        """
        return (input_ids != self.pad_token_id).float()
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through BERT4Rec model.
        
        Data Flow:
        1. Convert item IDs to embeddings
        2. Add positional information
        3. Pass through encoder layers (self-attention + feed-forward)
        4. Generate item predictions
        
        Args:
            input_ids: Item sequences (batch_size, seq_len)
            attention_mask: Mask for padding tokens (batch_size, seq_len)
            
        Returns:
            Item logits (batch_size, seq_len, item_vocab_size)
        """

        if attention_mask is None:
            # basically, it is a mask of the padded tokens (0 for padded hence ignored by attention)
            attention_mask = self.create_padding_mask(input_ids)
        
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        embedded = self.embeddings(input_ids) 
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        encoded = self.encoder(embedded, attention_mask)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        hidden = self.final_layer(encoded)          # (B, S, d_model)
        hidden = F.gelu(hidden)                     # (B, S, d_model)
        # self.item_embeddings.lut.weight: (vocab_size, d_model)
        # F.linear: (B, S, d_model) x (d_model, vocab_size) + (vocab_size,) -> (B, S, vocab_size)
        logits = F.linear(hidden, self.item_embeddings.lut.weight, self.output_bias)  # (B, S, vocab_size)
        
        return logits
    
    def predict_masked_items(self, input_ids: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           top_k: int = 10) -> Dict[str, torch.Tensor]:
        """
        Predict items for masked positions in the sequence.
        
        Purpose: During inference, predict what items should fill masked positions
        based on the surrounding context. This is the core recommendation functionality.
        
        Args:
            input_ids: Input sequences with [MASK] tokens (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and scores
        """
        # Get model predictions
        logits = self.forward(input_ids, attention_mask)
        
        # Find positions with [MASK] tokens
        mask_positions = (input_ids == self.mask_token_id)
        
        # Extract predictions only for masked positions
        masked_logits = logits[mask_positions]
        
        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(masked_logits, top_k, dim=-1)
        
        return {
            'logits': logits,
            'masked_logits': masked_logits,
            'top_k_items': top_k_indices,
            'top_k_scores': top_k_scores,
            'mask_positions': mask_positions
        }
    
    def compute_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked language modeling loss.
        
        Purpose: Train the model to predict masked items correctly. The loss is
        computed only for masked positions to focus learning on the recommendation task.
        
        Training Strategy:
        1. Take original sequence: [A, B, C, D, E]
        2. Create masked version: [A, [MASK], C, [MASK], E]
        3. Predict original items at masked positions
        4. Compute loss only for masked positions
        
        Args:
            input_ids: Input sequences with masks
            target_ids: Original sequences (ground truth)
            attention_mask: Attention mask
            loss_mask: Positions to compute loss for
            
        Returns:
            Cross-entropy loss
        """
        # Get model predictions
        logits = self.forward(input_ids, attention_mask)
        
        # Create loss mask if not provided (only masked positions)
        if loss_mask is None:
            loss_mask = (input_ids == self.mask_token_id)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, self.item_vocab_size)
        targets_flat = target_ids.view(-1)
        loss_mask_flat = loss_mask.view(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        masked_loss = loss * loss_mask_flat.float()
        
        # Average over masked positions
        return masked_loss.sum() / loss_mask_flat.sum().clamp(min=1)
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get learned item embeddings for analysis."""
        return self.item_embeddings.lut.weight
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'item_vocab_size': self.item_vocab_size,
                'max_seq_len': self.max_seq_len,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'pad_token_id': self.pad_token_id,
                'mask_token_id': self.mask_token_id,
            }
        }, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_bert4rec_model(item_vocab_size: int, **kwargs) -> BERT4Rec:
    """
    Factory function to create a BERT4Rec model with sensible defaults.
    
    Purpose: Provide an easy way to create models with good default parameters
    while allowing customization of important hyperparameters.
    
    Args:
        item_vocab_size: Number of unique items in your dataset
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Configured BERT4Rec model
    """
    defaults = {
        'max_seq_len': 200,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1
    }
    defaults.update(kwargs)
    return BERT4Rec(item_vocab_size=item_vocab_size, **defaults)
