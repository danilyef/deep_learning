import time

import torch


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark(fn, *args, runs=10):
    device = args[0].device.type
    fn(*args)  # warmup
    sync(device)

    start = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    sync(device)
    print(f"{device}: {(time.perf_counter() - start) / runs * 1000:.3f} ms")
