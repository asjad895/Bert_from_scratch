import torch
import torch.nn as nn
import numpy as np

# Embedding class: This class creates the token, position, and segment embeddings and normalizes them.
class Embedding(torch.nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        
        # Token embedding: maps each token in the vocabulary to a d_model-dimensional vector.
        # vocab_size: Size of the vocabulary (number of unique tokens in the corpus)
        # d_model: Dimensionality of the embedding vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding: Encodes the position of each token in the sequence (useful for transformers).
        # max_length: Maximum length of the input sequence.
        # d_model: Dimensionality of the embedding vectors.
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Segment embedding: Used for differentiating between different segments (e.g., in BERT for sentence A and B).
        # n_segments: Number of segments (usually 2 for sentence A and sentence B in BERT).
        # d_model: Dimensionality of the embedding vectors.
        self.segment_embedding = nn.Embedding(n_segments, d_model)

        # Layer normalization: Normalizes the output embedding to have zero mean and unit variance.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        # x: [batch_size, seq_len], token indices for the input sequence
        # seg: [batch_size, seq_len], segment indices for each token in the sequence

        seq_len = x.size(1)  # Get the sequence length (number of tokens in each input sequence)
        
        # Create position indices (for each position in the sequence)
        pos = torch.arange(seq_len, torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [batch_size, seq_len], repeat position indices for each batch
        
        # Sum of embeddings: token embedding + position embedding + segment embedding
        embedding = self.token_embedding(x) + self.position_embedding(pos) + self.segment_embedding(seg)

        # Apply layer normalization to the summed embeddings
        return self.norm(embedding)  # Output: [batch_size, seq_len, d_model]


# Function to create the attention mask for padding tokens (PAD tokens should not contribute to attention).
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()  # seq_q: [batch_size, len_q], seq_k: [batch_size, len_k]
    batch_size, len_k = seq_k.size()
    
    # Create a mask for padding tokens (PAD tokens are usually marked as '0')
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True for padding tokens
    # Expand the padding mask to match the shape [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


# Scaled Dot-Product Attention: Core attention mechanism where each token attends to all others in the sequence.
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: Query, K: Key, V: Value. All have shape [batch_size, n_heads, len_q, d_k] (or d_v).
        
        # Calculate the attention scores by performing a matrix multiplication of Query and Transposed Key
        # Scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # Normalization by sqrt(d_k)

        # Apply the attention mask (mask out the padding tokens by assigning a very low value to them)
        scores.masked_fill_(attn_mask, -1e9)  # Fills masked positions with a very small number to nullify them in the softmax
        
        # Apply softmax to the scores to get attention weights (probabilities)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]

        # Compute the context by applying the attention weights to the values
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        
        return scores, context, attn  # Return the raw scores, context (weighted sum of values), and attention weights


# Multi-Head Attention: This module implements multi-head attention by applying ScaledDotProductAttention multiple times.
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Linear projections for Q, K, and V. Each one produces a different set of projections per attention head.
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # Linear transformation for queries
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # Linear transformation for keys
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # Linear transformation for values

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, len_q, d_model], K: [batch_size, len_k, d_model], V: [batch_size, len_k, d_model]
        residual, batch_size = Q, Q.size(0)

        # Apply the linear transformations to Q, K, and V to get their projections
        # q_s: [batch_size, len_q, n_heads, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        
        # k_s: [batch_size, len_k, n_heads, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        
        # v_s: [batch_size, len_k, n_heads, d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # Repeat the attention mask for each head
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size, n_heads, len_q, len_k]

        # Get the context (attended values) from ScaledDotProductAttention
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        
        # Concatenate the contexts from all heads and project back to the original dimension
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)  # [batch_size, len_q, n_heads * d_v]
        
        # Output projection: map the concatenated context back to d_model dimensions
        output = nn.Linear(n_heads * d_v, d_model)(context)  # [batch_size, len_q, d_model]

        # Add the residual connection (skip connection) and apply layer normalization
        return nn.LayerNorm(d_model)(output + residual), attn  # Output: [batch_size, len_q, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # Feedforward network consists of two linear layers with ReLU activation between them
        self.fc1 = nn.Linear(d_model, d_ff)  # First linear layer (d_model -> d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)  # Second linear layer (d_ff -> d_model)
        self.activation = nn.GELU()  # GeLU activation function

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        The input is passed through the first linear layer, then through ReLU, and then through the second linear layer.
        """
        # Apply the first linear transformation, followed by ReLU activation
        x = self.activation(self.fc1(x))  # [batch_size, seq_len, d_ff]
        
        # Apply the second linear transformation (output shape: [batch_size, seq_len, d_model])
        x = self.fc2(x)  # [batch_size, seq_len, d_model]
        return x


# Encoder Layer: This is a single layer in the transformer encoder, combining multi-head self-attention and position-wise feedforward networks.
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # Multi-head self-attention mechanism
        self.enc_self_attn = MultiHeadAttention()
        # Position-wise feedforward network (FFN) applied to each position independently
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # Apply multi-head attention to the inputs
        # enc_inputs: [batch_size, len_q, d_model], enc_self_attn_mask: [batch_size, len_q, len_k]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # Q=K=V

        # Apply position-wise feedforward network
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, len_q, d_model]

        # Return the encoder outputs and the attention weights
        return enc_outputs, attn

