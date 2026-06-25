import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



################ Dataset ################

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader




################ Architecture Elements ################


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/ torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        # self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.register_buffer('cache_k', None, persistent=False) # persistent - whether the buffer is part of the module’s state_dict
        self.register_buffer('cache_v', None, persistent=False)  # persistent - whether the buffer is part of the module’s state_dict
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        batch, num_tokens, dim = x.shape

        if use_cache:
            assert num_tokens <= self.window_size, (
                f"Input chunk size ({num_tokens}) exceeds KV cache window size ({self.window_size}). "
            )

        queries = self.W_query(x)
        values_new = self.W_value(x)
        keys_new = self.W_key(x)

        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(batch, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1,2)
        values_new = values_new.transpose(1,2)
        keys_new = keys_new.transpose(1,2)

        if use_cache:

            if self.cache_k is None or self.cache_k.size(0) != batch:
                self.cache_k = torch.zeros(batch, self.num_heads, self.window_size, self.head_dim, device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0

            # Discard oldest tokens, if overflow
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens  - self.window_size
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone() # Shift cache left by `overflow` tokens.
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone() # Example: if overflow = 5, tokens [5:] are moved to the front and the first 5 (oldest) tokens are discarded.
                self.ptr_cur -= overflow
                
                # overflow = 3
                # index:   0 1 2 3 4 5 6 7 8 9
                # values: [A B C D E F G H I J]
                # [:-3]: [A B C D E F G]
                # [:3]:   [D E F G H I J]   
                # [A B C D E F G] = [D E F G H I J]   
                # result: [D E F G H I J ? ? ?]   ? tokens are waiting to e overwritten
            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = values_new
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]

        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        attn_scores = queries @ keys.transpose(2, 3) # (batch, heads, Q, K)


        K = attn_scores.size(-1) # number of Keys
        if num_tokens == K:
            # No cache → use the pre‑baked triangular mask slice
            casual_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # cached
            offset = K - num_tokens # number of tokens already in cache before this chunk
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1) # (num_tokens, 1) -> query token positions. row 0 → first new token, row 1 → second new token
            col_idx = torch.arange(K, device=x.device).unsqueeze(0) # (1, K) # key positions, col 0 → first cached token, col 5 → newest token
            # unsqeezes -> for broadcasting (num_tokens, K)
            casual_mask = row_idx + offset < col_idx # True where j > i+offset: mask future tokens; Query index i can see keys up to (offset + i)
            # offset to catch the correct index: cache have older indicies, query newer -> missmatch.

            # Example: num_tokens=2, K=8, offset=6
            #
            #   row_idx          [[0], [1]]          (num_tokens, 1)
            #   col_idx          [[0 1 2 3 4 5 6 7]] (1, K)
            #   row_idx + offset [[6], [7]]
            #
            #   row_idx + offset < col_idx:
            #     6 < [0..7] → F F F F F F F T
            #     7 < [0..7] → F F F F F F F F
            #
            #   Q0 (pos 6) attends to keys ≤6, Q1 (pos 7) attends to keys ≤7

        attn_scores.masked_fill_(casual_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1,2)
        context_vector = context_vector.contiguous().view(batch, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_cur = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x,use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ptr_current_pos = 0

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.head_out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.kv_window_size = cfg["kv_window_size"]  if "kv_window_size" in cfg else cfg["context_length"]

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if use_cache:
            context_length = self.pos_emb.num_embeddings
            assert self.ptr_current_pos + seq_len <= context_length, (
                f"Position embedding overflow. Want to read {self.ptr_current_pos + seq_len} which excceded size of {context_length}"
            )
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(seq_len, device = in_idx.device, dtype=torch.long)

        pos_embeds = self.pos_emb(pos_ids)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        logits = self.head_out(x)

        return logits 

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0       
    

################ Text Generation ################

def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings
    kv_window_size = model.kv_window_size
    
    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            input_tokens = idx[:, -ctx_len:]
            input_tokens_length = input_tokens.size(1)


            # prefill to handle input_tokens_length > kv_window_size
            # It's needed when: prompt length > KV window size
            for i in range(0, input_tokens_length, kv_window_size):
                chunk = input_tokens[:, i:i + kv_window_size]
                logits = model(chunk, use_cache=True)

            # can't generate more than ctx_len of result
            # due to the limitation of position embedding
            max_generable = ctx_len - input_tokens_length
            max_new_tokens = min(max_new_tokens, max_generable)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=use_cache)

        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=use_cache)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


################ Main Function ################
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False,        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Hello, I am"
    
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print("\nInput text:", start_context)
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()