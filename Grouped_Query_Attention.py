import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_queries = config.n_head        # Total Query heads
        self.n_kv_groups = config.n_kv_groups # Number of KV heads (G)
        self.n_rep = self.n_queries // self.n_kv_groups
        
        self.head_dim = config.n_embd // config.n_head
        self.embed_dim = config.n_embd
        
        # Projections: Note the smaller sizes for K and V
        self.q_proj = nn.Linear(self.embed_dim, self.n_queries * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.n_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.n_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x, is_causal=True):
        batch, seq_len, _ = x.shape
        
        # 1. Project Q, K, V
        q = self.q_proj(x) # (B, T, n_queries * head_dim)
        k = self.k_proj(x) # (B, T, n_kv_groups * head_dim)
        v = self.v_proj(x) # (B, T, n_kv_groups * head_dim)

        # 2. Reshape for attention
        q = q.view(batch, seq_len, self.n_queries, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # 3. Repeat KV heads to match number of Query heads
        # This is where the 'Grouped' magic happens
        if self.n_rep > 1:
            # We use repeat_interleave so that Q heads 0,1,2,3... map to K head 0
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 4. Efficient Attention (FlashAttention compatible)
        # Scale is applied internally: 1 / sqrt(head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=is_causal,
            dropout_p=0.0 if not self.training else 0.1
        )

        # 5. Combine heads and project out
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)
