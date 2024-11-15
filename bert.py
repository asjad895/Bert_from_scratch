import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_parts import Embedding,EncoderLayer,get_attn_pad_mask
import math

def gelu(x):
    # GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# BERT model definition
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.n_layers = 12
        self.d_model = 768
        # Embedding Layer: This combines token, position, and segment embeddings.
        self.embedding = Embedding()

        # Encoder Layers: Stack of N transformer encoder layers
        # Each encoder layer consists of self-attention and position-wise feedforward networks
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(self.n_layers)])

        # Feedforward network for the final pooled output
        self.fc = nn.Linear(self.d_model, self.d_model)
        
        # Activation function for the pooled output
        self.activ1 = nn.Tanh()

        # Feedforward network for masked language modeling (LM)
        self.linear = nn.Linear(self.d_model, self.d_model)

        # GELU (Gaussian Error Linear Unit) activation function for the masked LM output
        self.activ2 = gelu

        # Layer normalization applied after the masked language model output
        self.norm = nn.LayerNorm(self.d_model)

        # Classifier head for the output (classification task: 2 classes)
        self.classifier = nn.Linear(self.d_model, 2)

        # Decoder for the language model (LM), shared with the embedding layer
        embed_weight = self.embedding.tok_embed.weight  # Get the token embedding weights
        n_vocab, n_dim = embed_weight.size()  # n_vocab: size of the vocabulary, n_dim: dimensionality of the embedding
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)  # Decoder projects d_model to vocab size
        
        # Initialize decoder weights with the token embedding weights
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))  # Bias for the decoder output

    def forward(self, input_ids, segment_ids, masked_pos):
        """
        input_ids: [batch_size, seq_len], token indices in the input sequence.
        segment_ids: [batch_size, seq_len], segment indices (used in tasks like sentence-pair classification).
        masked_pos: [batch_size, max_pred], positions of the tokens that are masked for MLM.
        
        Returns:
        logits_lm: [batch_size, max_pred, vocab_size], logits for masked language modeling (MLM).
        logits_clsf: [batch_size, 2], logits for classification task.
        """

        # Step 1: Get the embedding of input tokens (token embeddings + position embeddings + segment embeddings)
        output = self.embedding(input_ids, segment_ids)

        # Step 2: Create attention mask for padding (PAD tokens should be masked)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        # Step 3: Pass the embeddings through each encoder layer (transformer layers)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)  # [batch_size, seq_len, d_model]

        # Step 4: Use the output from the first token ([CLS] token) for classification
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model] from the [CLS] token (index 0)
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] logits for classification (binary)

        # Step 5: Get the embeddings for the masked tokens
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # Expand to match the dimensionality: [batch_size, max_pred, d_model]

        # Step 6: Gather the embeddings for the masked tokens based on the positions
        h_masked = torch.gather(output, 1, masked_pos)  # [batch_size, max_pred, d_model] (gather masked tokens)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))  # Apply linear transformation, GELU, and layer normalization

        # Step 7: Decoder for the masked language modeling (LM) task
        logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf  # Return logits for both MLM and classification tasks
