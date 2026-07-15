import math
import torch

from einops import einsum, rearrange
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    
    # Calculate attention scores.
    attention_scores = einsum(query, key, 'batch num_heads seq_len1 head_dims, batch num_heads seq_len2 head_dims -> batch num_heads seq_len1 seq_len2')
    # Scaled dot product of attention scores.
    attention_scores = attention_scores * (1.0 / math.sqrt(key.size(-1)))
    
    # print("Attention scores before masking", tuple(attention_scores.shape))
    # print(attention_scores[0, 0, :5, :5])

    T = attention_scores.size(-1)
    causal_mask = torch.triu(torch.ones(T, T, device=attention_scores.device), diagonal=1) * -10000.0
    attention_scores = attention_scores + causal_mask
    attention_scores = attention_scores + attention_mask
    # attention_scores = attention_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

    # print("Attention scores after masking", tuple(attention_scores.shape))
    # print(attention_scores[0, 0, :5, :5])

    softmax_scores = F.softmax(attention_scores, dim=-1)
    # print("softmax_scores", tuple(softmax_scores.shape))
    # print(softmax_scores[0, 0, :5, :5])
    # print("value", tuple(value.shape))
    # print(value[0, 0, :5, :5])

    # Apply dropout to the normalized attention scores (per the note in __init__).
    softmax_scores = self.dropout(softmax_scores)

    # print(f"softmax scores shape: {softmax_scores.shape}")
    weighted_attention_values = softmax_scores @ value

    # print("weighted_attention_values", tuple(weighted_attention_values.shape))
    # print(weighted_attention_values[0, 0, :5, :5])
    weighted_attention_values = rearrange(weighted_attention_values, "batch num_heads seq_len head_dims -> batch seq_len (num_heads head_dims)")

    return weighted_attention_values


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
