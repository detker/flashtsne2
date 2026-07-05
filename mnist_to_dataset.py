"""Convert MNIST (train+test, 70k x 784) to the raw float32 .dat format
produced by generate_datasets.py: row-major, no header, n = filesize / (4*dim).

Usage: python mnist_to_dataset.py
Writes datasets/DATASET_784_10_mnist.dat and datasets/mnist_labels.txt
(labels are not read by hpc_app - they are for coloring the final embedding).
"""
import gzip
import urllib.request
from pathlib import Path

import numpy as np

OUT_DIR = Path("datasets")
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
IMAGES = ["train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
LABELS = ["train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def fetch(name: str, cache_dir: Path) -> Path:
    path = cache_dir / name
    if not path.exists():
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(MIRROR + name, path)
    return path


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    return np.frombuffer(data, np.uint8, offset=16).reshape(n, rows * cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data, np.uint8, offset=8, count=n)


def main():
    cache = OUT_DIR / "mnist_raw"
    cache.mkdir(parents=True, exist_ok=True)

    x = np.vstack([read_idx_images(fetch(f, cache)) for f in IMAGES])
    x = x.astype(np.float32) / 255.0
    y = np.concatenate([read_idx_labels(fetch(f, cache)) for f in LABELS])

    out = OUT_DIR / "DATASET_784_10_mnist.dat"
    x.tofile(out)
    np.savetxt(OUT_DIR / "mnist_labels.txt", y, fmt="%d")
    print(f"{out}: {x.shape[0]:,} x {x.shape[1]} float32 "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"labels -> {OUT_DIR / 'mnist_labels.txt'}")


if __name__ == "__main__":
    main()
