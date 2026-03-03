# pip install flash-attn
from flash_attn import flash_attn_qkvpacked_func

class FlashAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # We pack QKV into one projection for maximum throughput
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        # FlashAttention expects input in B, T, C
        # and it performs its own internal reshaping.
        qkv = self.Wqkv(x)
        
        # Reshape to (Batch, SeqLen, 3, NumHeads, HeadDim)
        # This specific layout is required by the Flash kernel
        new_shape = (x.shape[0], x.shape[1], 3, self.n_head, self.n_embd // self.n_head)
        qkv = qkv.view(*new_shape)

        # flash_attn_qkvpacked_func:
        # 1. No need to manually transpose heads (saves memory)
        # 2. Computes softmax + weighted sum in SRAM
        # 3. Supports causal masking natively
        out = flash_attn_qkvpacked_func(
            qkv, 
            dropout_p=0.0, 
            softmax_scale=None, 
            causal=True
        )

        # Flatten heads back to embedding dimension
        out = out.view(x.shape[0], x.shape[1], self.n_embd)
        return self.out_proj(out)
