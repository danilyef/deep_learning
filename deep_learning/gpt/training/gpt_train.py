import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    data_dir: str = '/home/dan_yef/gutenberg_preprocessed'
    batch_size: int = 8

    # Optimization
    num_epochs: int = 2
    learning_rate: float = 5e-4       
    pct_start: float = 0.03          
    weight_decay: float = 0.01
    gradient_clipping: bool = True

    # Evaluation
    eval_freq: int = 1000
    eval_iter: int = 20

    # Checkpointing
    output_dir: str = "/home/dan_yef/model_checkpoints"
    save_ckpt_freq: int = 100_000


@dataclass
class ModelConfig:
    vocab_size: int = 50257           # Vocabulary size
    context_length: int = 1024        # Context length
    emb_dim: int = 768                # Embedding dimension
    n_heads: int = 12                 # Number of attention heads
    n_layers: int = 12                # Number of layers
    drop_rate: float = 0.1            # Dropout rate
    qkv_bias: bool = False            # Query-key-value bias


class GPTDataset(Dataset):

    def __init__(self, path_bin_file, max_length, stride):
        self.max_length = max_length
        self.stride = stride
        self.all_tokens = np.memmap(path_bin_file, dtype=np.uint16, mode='r')

    def __len__(self):
        return math.ceil((len(self.all_tokens) - self.max_length) / self.stride)
    
    def __getitem__(self, index):
        inputs = self.all_tokens[index * self.stride: index * self.stride + self.max_length].astype(np.int64)
        targets = self.all_tokens[index * self.stride + 1: index * self.stride + self.max_length + 1].astype(np.int64)
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

def create_dataloader(path_bin_file, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0, num_samples=None):
    dataset = GPTDataset(path_bin_file, max_length, stride)
    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        indices = torch.linspace(0, len(dataset) - 1, num_samples).long().tolist()
        dataset = Subset(dataset, indices)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader



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
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim)
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
            causal_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # cached
            offset = K - num_tokens # number of tokens already in cache before this chunk
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1) # (num_tokens, 1) -> query token positions. row 0 → first new token, row 1 → second new token
            col_idx = torch.arange(K, device=x.device).unsqueeze(0) # (1, K) # key positions, col 0 → first cached token, col 5 → newest token
            # unsqeezes -> for broadcasting (num_tokens, K)
            causal_mask = row_idx + offset < col_idx # True where j > i+offset: mask future tokens; Query index i can see keys up to (offset + i)
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

        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        
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
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

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
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ptr_current_pos = 0

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.head_out = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.kv_window_size = getattr(cfg, "kv_window_size", cfg.context_length)

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
    
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if(len(data_loader) == 0):
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, targeg_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, targeg_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, scheduler, training_config):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(training_config.num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            with torch.autocast(device_type=training_config.device, dtype=torch.bfloat16):
                loss = calc_loss_batch(input_batch, target_batch, model, training_config.device)
            loss.backward()

            if training_config.gradient_clipping:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step +=1

            if global_step % training_config.eval_freq == 0:
                with torch.no_grad():
                    model.eval()
                    train_loss = calc_loss_loader(train_loader, model, training_config.device, num_batches = training_config.eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, training_config.device, num_batches = training_config.eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                model.train()

            if global_step % training_config.save_ckpt_freq == 0 and global_step != 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, Path(training_config.output_dir) / f"checkpoint_{global_step}.pt" 
                )

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, Path(training_config.output_dir) / f"last_checkpoint.pt" 
    )
    
    
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room


def main(model_config, training_config):
    
    torch.manual_seed(123)

    model = GPTModel(model_config)
    model.to(device=training_config.device)

    train_file_path = Path(training_config.data_dir) / "train.bin"
    valid_file_path = Path(training_config.data_dir) / "val.bin"


    train_loader = create_dataloader(
        train_file_path,
        batch_size=training_config.batch_size,
        max_length=model_config.context_length,
        stride=model_config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader(
        valid_file_path,
        batch_size=training_config.batch_size,
        max_length=model_config.context_length,
        stride=model_config.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        num_samples=training_config.eval_iter * training_config.batch_size
    )


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=training_config.learning_rate, pct_start= training_config.pct_start, 
        steps_per_epoch=len(train_loader), epochs=training_config.num_epochs, cycle_momentum=False          
    )

    # Train model
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, scheduler,
        training_config=training_config,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    MODEL_CONFIG = ModelConfig()
    TRAINING_CONFIG = TrainConfig()

    train_losses, val_losses, tokens_seen, model = main(MODEL_CONFIG, TRAINING_CONFIG)
    epochs_tensor = torch.linspace(0, TRAINING_CONFIG.num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")


