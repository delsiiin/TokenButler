import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.profiler import profile, ProfilerActivity

# Class and functions
class IndexCache:
    def __init__(self, B: int, H: int, max_len: int, device):
        self.B = B
        self.H = H
        self.max_len = max_len
        self.device = device
        self._batch_indices = torch.arange(B, device=device).view(-1, 1, 1).expand(-1, H, max_len)
        self._head_indices = torch.arange(H, device=device).view(1, -1, 1).expand(B, -1, max_len)

    def get_indices(self, seq_len: int):
        return (
            self._batch_indices[:, :, :seq_len],
            self._head_indices[:, :, :seq_len]
        )

def gather_kv_for_indices_batch(key_states, value_states, indices_tensor):
    indices_expanded = indices_tensor.unsqueeze(-1).expand(-1, -1, -1, key_states.size(-1))
    sub_k = torch.gather(key_states, 2, indices_expanded)
    sub_v = torch.gather(value_states, 2, indices_expanded)
    return sub_k, sub_v

@torch.compile
def sdp_attention_flash(query_states, key_states, value_states, num_key_value_groups):
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    return F.scaled_dot_product_attention(query_states, key_states, value_states)

@torch.compile
def sdp_attention_flash_sparsity(query_states, key_states, value_states, indices_tensor, num_key_value_groups):
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    sub_k, sub_v = gather_kv_for_indices_batch(key_states, value_states, indices_tensor)
    return F.scaled_dot_product_attention(query_states, sub_k, sub_v)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)

def main():
    # Parameters and configurations
    B, H, H_kv, hidden_dim = 1, 32, 8, 128
    num_key_value_groups = H // H_kv
    seq_len_list = [2048, 4096, 8192, 16384, 32768, 32768*2, 131072]
    # seq_len_list = [2048, 4096, 8192, 32768, 131072, 262144, 524288]
    num_warmup, num_runs = 10, 100

    DEVICE = torch.device("cuda")
    os.makedirs("perf_results", exist_ok=True)

    # Data for plotting
    seq_len_axis = []
    full_attention_latency = []
    sparse_attention_latency = []

    for seq_len in seq_len_list:
        max_len = seq_len // 2
        seq_len_axis.append(seq_len)

        # Prepare tensors
        query_states = torch.randn(B, H, 1, hidden_dim, device=DEVICE)
        key_states = torch.randn(B, H_kv, seq_len, hidden_dim, device=DEVICE)
        value_states = torch.randn(B, H_kv, seq_len, hidden_dim, device=DEVICE)
        indices_tensor = torch.randint(0, seq_len, (B, H, max_len), device=DEVICE)
        indices_tensor = torch.sort(indices_tensor, dim=-1).values

        # Measure FullAttention latency
        for _ in range(num_warmup):
            _ = sdp_attention_flash(query_states, key_states, value_states, num_key_value_groups)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            _ = sdp_attention_flash(query_states, key_states, value_states, num_key_value_groups)
        torch.cuda.synchronize()
        total_time_original = time.time() - start_time
        full_attention_latency.append(total_time_original / num_runs * 1e6)  # Convert to microseconds

        # Measure SparseAttention latency
        for _ in range(num_warmup):
            _ = sdp_attention_flash_sparsity(query_states, key_states, value_states, indices_tensor, num_key_value_groups)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            _ = sdp_attention_flash_sparsity(query_states, key_states, value_states, indices_tensor, num_key_value_groups)
        torch.cuda.synchronize()
        total_time_sub = time.time() - start_time
        sparse_attention_latency.append(total_time_sub / num_runs * 1e6)  # Convert to microseconds

        # Profiling
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler(f"perf_results/profile_seq_{seq_len}")) as prof:
            for _ in range(10):  # Profile 10 runs
                _ = sdp_attention_flash(query_states, key_states, value_states, num_key_value_groups)
                _ = sdp_attention_flash_sparsity(query_states, key_states, value_states, indices_tensor, num_key_value_groups)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Plot results
    plt.figure(figsize=(10, 8))
    bar_width = 0.35  # Width of each bar
    x = range(len(seq_len_list))

    plt.bar([i - bar_width / 2 for i in x], full_attention_latency, bar_width, label="FullAttention", color="steelblue")
    plt.bar([i + bar_width / 2 for i in x], sparse_attention_latency, bar_width, label="SparseAttention", color="darkorange")

    plt.xlabel("Sequence Length", fontsize=22)
    plt.ylabel("Latency (μs)", fontsize=22)
    plt.xticks(x, [f"{seq}" for seq in seq_len_list], fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig("perf_results/attn_only_latency.pdf")
    print("Saved latency graph and profiles in 'perf_results/' directory.")

if __name__ == "__main__":
    main()
