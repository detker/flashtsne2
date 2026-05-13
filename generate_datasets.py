#!/usr/bin/env python3
"""Generate 2D clustering datasets as raw float32 binary files."""

import numpy as np
from pathlib import Path

OUT_DIR = Path("datasets")
OUT_DIR.mkdir(exist_ok=True)

DIM = 2
RNG = np.random.default_rng(42)

datasets = [
    # (n_points, k, description)
    (100_000,       2, "100k_2"),
    (100_000,       4, "100k_4"),
    (100_000,       8, "100k_8"),
    (250_000,       3, "250k_3"),
    (250_000,       6, "250k_6"),
    (500_000,       2, "500k_2"),
    (500_000,       5, "500k_5"),
    (500_000,       8, "500k_8"),
    (1_000_000,     3, "1M_3"),
    (1_000_000,     8, "1M_8")
]


def generate_blobs(n: int, k: int) -> np.ndarray:
    """Generate n messy 2D points that aren't perfectly separable.

    Realism tricks:
    - Unequal cluster weights (some clusters much bigger)
    - Overlapping clusters (close centers + wide spreads)
    - Elongated clusters via random covariance
    - Uniform background noise (5-15% of points)
    """
    # random cluster weights — some clusters dominate
    weights = RNG.dirichlet(np.ones(k) * 0.5)

    # centers can be close together → overlap
    centers = RNG.uniform(-30, 30, size=(k, DIM)).astype(np.float64)

    # how much background noise
    noise_frac = RNG.uniform(0.05, 0.15)
    n_noise = int(n * noise_frac)
    n_cluster = n - n_noise
    counts = (weights * n_cluster).astype(int)
    counts[-1] = n_cluster - counts[:-1].sum()  # fix rounding

    chunks = []
    for c in range(k):
        # random 2x2 covariance for elongated/rotated clusters
        angle = RNG.uniform(0, np.pi)
        sx, sy = RNG.uniform(1.0, 8.0, size=2)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        cov = rot @ np.diag([sx**2, sy**2]) @ rot.T

        blob = RNG.multivariate_normal(centers[c], cov, size=counts[c]).astype(np.float32)
        chunks.append(blob)

    # uniform background noise spanning the data range
    lo = centers.min(axis=0) - 20
    hi = centers.max(axis=0) + 20
    noise = RNG.uniform(lo, hi, size=(n_noise, DIM)).astype(np.float32)
    chunks.append(noise)

    data = np.vstack(chunks)
    RNG.shuffle(data)
    return data


CHUNK = 10_000_000  # write 10M points at a time for the big dataset

for n, k, tag in datasets:
    path = OUT_DIR / f"DATASET_{DIM}_{k}_{tag}.dat"
    size_gb = n * DIM * 4 / 1e9
    print(f"Generating {path.name}  n={n:>13,}  k={k}  ({size_gb:.2f} GB) ...", flush=True)

    if n <= CHUNK:
        data = generate_blobs(n, k)
        data.tofile(path)
    else:
        # Stream large datasets to disk in chunks to avoid OOM
        # Same messy generation: unequal weights, elongated clusters, noise
        weights = RNG.dirichlet(np.ones(k) * 0.5)
        centers = RNG.uniform(-30, 30, size=(k, DIM)).astype(np.float64)
        noise_frac = RNG.uniform(0.05, 0.15)

        # precompute covariance matrices
        covs = []
        for c in range(k):
            angle = RNG.uniform(0, np.pi)
            sx, sy = RNG.uniform(1.0, 8.0, size=2)
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            covs.append(rot @ np.diag([sx**2, sy**2]) @ rot.T)

        lo = centers.min(axis=0) - 20
        hi = centers.max(axis=0) + 20

        remaining = n
        with open(path, "wb") as f:
            while remaining > 0:
                batch = min(CHUNK, remaining)
                n_noise = int(batch * noise_frac)
                n_cluster = batch - n_noise

                # weighted cluster assignment
                cluster_ids = RNG.choice(k, size=n_cluster, p=weights)
                points = np.empty((n_cluster, DIM), dtype=np.float32)
                for c in range(k):
                    mask = cluster_ids == c
                    count = mask.sum()
                    if count > 0:
                        points[mask] = RNG.multivariate_normal(
                            centers[c], covs[c], size=count).astype(np.float32)

                # uniform noise
                noise = RNG.uniform(lo, hi, size=(n_noise, DIM)).astype(np.float32)
                chunk_data = np.vstack([points, noise])
                RNG.shuffle(chunk_data)
                chunk_data.tofile(f)

                remaining -= batch
                if remaining % (100 * CHUNK) == 0 and remaining > 0:
                    print(f"  ... {remaining:,} points remaining", flush=True)

    print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")
