#!/usr/bin/env python3
"""
Synthetic-image benchmark for MMDetection models.

Usage examples:
# CPU, 固定线程数以获得稳定结果
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/benchmark_synthetic.py \
    configs/xxx/xxx_config.py checkpoints/xxx.pth --device cpu --num-iters 200 --warmup 20 --height 800 --width 1333 --threads 4

# GPU
python tools/benchmark_synthetic.py configs/xxx/xxx_config.py checkpoints/xxx.pth --device cuda:0 --num-iters 500 --warmup 50 --height 800 --width 1333
"""
import argparse
import time
import statistics
import numpy as np
import torch
from mmengine.config import Config

from mmdet.apis import init_detector, inference_detector

def parse_args():
    parser = argparse.ArgumentParser('Synthetic benchmark for MMDet')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='device, e.g. "cpu" or "cuda:0"')
    parser.add_argument('--num-iters', type=int, default=200, help='number of measured inferences')
    parser.add_argument('--warmup', type=int, default=10, help='number of warmup runs')
    parser.add_argument('--height', type=int, default=800, help='image height')
    parser.add_argument('--width', type=int, default=1333, help='image width')
    parser.add_argument('--channels', type=int, default=3, help='image channels')
    parser.add_argument('--dtype', choices=['uint8', 'float32'], default='uint8', help='image dtype to generate')
    parser.add_argument('--threads', type=int, default=None, help='torch.set_num_threads (CPU only)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def make_random_image(h, w, c, dtype='uint8', seed=None):
    if seed is not None:
        np.random.seed(seed)
    if dtype == 'uint8':
        # typical image content range [0,255], HWC
        img = np.random.randint(0, 256, size=(h, w, c), dtype=np.uint8)
    else:
        # float32 in 0..1
        img = (np.random.rand(h, w, c).astype(np.float32))
    return img

def main():
    args = parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print('Initializing model...')
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Generate synthetic images list
    total_runs = args.warmup + args.num_iters
    imgs = []
    for i in range(total_runs):
        # optionally vary seed to avoid cache effects; otherwise identical images are okay too
        imgs.append(make_random_image(args.height, args.width, args.channels, dtype=args.dtype, seed=args.seed + i))

    # Warmup
    print(f'Warmup {args.warmup} iters...')
    for i in range(args.warmup):
        img = imgs[i]
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        _ = inference_detector(model, img)

    # Measured runs
    print(f'Measuring {args.num_iters} iters on device {args.device} ...')
    latencies = []
    for i in range(args.warmup, total_runs):
        img = imgs[i]
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = inference_detector(model, img)
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    total_time = sum(latencies)
    images = len(latencies)
    ips = images / total_time if total_time > 0 else float('inf')
    lat_ms = [l * 1000.0 for l in latencies]

    print('==== Synthetic Benchmark result ====')
    print(f'Device: {args.device}')
    print(f'Image size: {args.height}x{args.width}x{args.channels}')
    print(f'Images measured: {images}')
    print(f'Total measured time (s): {total_time:.4f}')
    print(f'FPS (images/sec): {ips:.2f}')
    print(f'Latency mean (ms): {statistics.mean(lat_ms):.2f}')
    print(f'Latency median (ms): {statistics.median(lat_ms):.2f}')
    print(f'Latency p95 (ms): {sorted(lat_ms)[int(0.95*len(lat_ms))-1]:.2f}')
    print(f'Latency min/max (ms): {min(lat_ms):.2f}/{max(lat_ms):.2f}')
    print('====================================')

if __name__ == '__main__':
    main()
