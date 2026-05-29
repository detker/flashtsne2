import numpy as np
from pathlib import Path

OUT_DIR = Path("datasets")
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

datasets = [
    # (n_points, k, dim, description)
    (100_000,       2,  2, "100k_2"),
    (100_000,       4,  2, "100k_4"),
    (100_000,       8,  2, "100k_8"),
    (250_000,       3,  2, "250k_3"),
    (250_000,       6,  2, "250k_6"),
    (500_000,       2,  2, "500k_2"),
    (500_000,       5,  2, "500k_5"),
    (500_000,       8,  2, "500k_8"),
    (1_000_000,     3,  2, "1M_3"),
    (1_000_000,     8,  2, "1M_8"),
    (100_000,       8, 10, "100k_8_d10"),
    (250_000,       6, 50, "250k_6_d50"),
]


def random_covariance(dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(RNG.standard_normal((dim, dim)))
    variances = RNG.uniform(1.0, 8.0, size=dim) ** 2
    return q @ np.diag(variances) @ q.T


def generate_blobs(n: int, k: int, dim: int) -> np.ndarray:
    weights = RNG.dirichlet(np.ones(k) * 0.5)

    centers = RNG.uniform(-30, 30, size=(k, dim)).astype(np.float64)

    noise_frac = RNG.uniform(0.05, 0.15)
    n_noise = int(n * noise_frac)
    n_cluster = n - n_noise
    counts = (weights * n_cluster).astype(int)
    counts[-1] = n_cluster - counts[:-1].sum()  # fix rounding

    chunks = []
    for c in range(k):
        cov = random_covariance(dim)
        blob = RNG.multivariate_normal(centers[c], cov, size=counts[c]).astype(np.float32)
        chunks.append(blob)

    lo = centers.min(axis=0) - 20
    hi = centers.max(axis=0) + 20
    noise = RNG.uniform(lo, hi, size=(n_noise, dim)).astype(np.float32)
    chunks.append(noise)

    data = np.vstack(chunks)
    RNG.shuffle(data)
    return data


CHUNK = 10_000_000 

for n, k, dim, tag in datasets:
    path = OUT_DIR / f"DATASET_{dim}_{k}_{tag}.dat"
    size_gb = n * dim * 4 / 1e9
    print(f"Generating {path.name}  n={n:>13,}  k={k}  dim={dim}  ({size_gb:.2f} GB) ...", flush=True)

    if n <= CHUNK:
        data = generate_blobs(n, k, dim)
        data.tofile(path)
    else:
        weights = RNG.dirichlet(np.ones(k) * 0.5)
        centers = RNG.uniform(-30, 30, size=(k, dim)).astype(np.float64)
        noise_frac = RNG.uniform(0.05, 0.15)

        covs = [random_covariance(dim) for _ in range(k)]

        lo = centers.min(axis=0) - 20
        hi = centers.max(axis=0) + 20

        remaining = n
        with open(path, "wb") as f:
            while remaining > 0:
                batch = min(CHUNK, remaining)
                n_noise = int(batch * noise_frac)
                n_cluster = batch - n_noise

                cluster_ids = RNG.choice(k, size=n_cluster, p=weights)
                points = np.empty((n_cluster, dim), dtype=np.float32)
                for c in range(k):
                    mask = cluster_ids == c
                    count = mask.sum()
                    if count > 0:
                        points[mask] = RNG.multivariate_normal(
                            centers[c], covs[c], size=count).astype(np.float32)

                noise = RNG.uniform(lo, hi, size=(n_noise, dim)).astype(np.float32)
                chunk_data = np.vstack([points, noise])
                RNG.shuffle(chunk_data)
                chunk_data.tofile(f)

                remaining -= batch
                if remaining % (100 * CHUNK) == 0 and remaining > 0:
                    print(f"  ... {remaining:,} points remaining", flush=True)

    print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")
