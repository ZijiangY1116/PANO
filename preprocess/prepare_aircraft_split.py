# Copyright 2026 Zijiang Yang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Apply the experimental first-140 / last-10 aircraft split and training statistics."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm


def case_moments(path):
    with np.load(path, allow_pickle=False) as data:
        result = []
        n_points = len(data["coords"])
        for key, width in (("condition", 10), ("sol", 6)):
            values = data[key].astype(np.float64)
            mean = values.mean(axis=0)
            m2 = np.square(values - mean).sum(axis=0)
            result.append((mean, m2))
    return n_points, result


def compute_stats(paths, workers):
    count = 0
    means = [np.zeros(10, dtype=np.float64), np.zeros(6, dtype=np.float64)]
    m2s = [np.zeros_like(mean) for mean in means]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, moments in tqdm(
            pool.map(case_moments, paths), total=len(paths), desc="Training statistics"
        ):
            total = count + n
            for i, (mean, m2) in enumerate(moments):
                delta = mean - means[i]
                m2s[i] += m2 + delta * delta * (count * n / total)
                means[i] += delta * (n / total)
            count = total
    stats = {}
    for i, key in enumerate(("condition", "sol")):
        stats[key + "_mean"] = means[i].astype(np.float32)
        stats[key + "_std"] = (np.sqrt(m2s[i] / count) + 1e-8).astype(np.float32)
    return stats, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, default="dataset/aircraft")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    manifest_path = os.path.join(args.input_dir, "manifest.json")
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    metadata = manifest["case_metadata"]
    expected = [
        row["case_name"]
        for row in sorted(
            metadata,
            key=lambda row: (
                row["shape_id"],
                row["mach"],
                row["alpha_deg"],
                row["beta_deg"],
            ),
        )
    ]
    cases = manifest["all_cases"]
    if len(cases) != 150 or len(set(cases)) != 150 or cases != expected:
        raise ValueError(
            "Expected 150 distinct cases in geometry/flight numeric order."
        )
    train, test = cases[:140], cases[140:]
    by_name = {row["case_name"]: row for row in metadata}
    train_shapes = sorted({by_name[name]["shape_id"] for name in train})
    test_shapes = sorted({by_name[name]["shape_id"] for name in test})
    # Re-running this same split is allowed; another existing split is preserved.
    stats, count = compute_stats(
        [os.path.join(args.input_dir, "cases", f"{name}.npz") for name in train],
        args.workers,
    )
    manifest.update(
        train_full=train,
        train=train,
        val=[],
        test=test,
        split_status="configured",
        statistics_status="training_only",
        split_definition={
            "name": "first140_last10",
            "official": False,
            "ordering": "numeric shape_id, mach, alpha_deg, beta_deg (matches raw result.csv)",
            "train_shapes": train_shapes,
            "test_shapes": test_shapes,
        },
        statistics={
            "source_split": "train",
            "n_cases": len(train),
            "n_points": count,
            "weighting": "point",
            "variance": "population",
            "std_epsilon": 1e-8,
            "accumulator_dtype": "float64",
            "stored_dtype": "float32",
        },
    )
    stats_path = os.path.join(args.input_dir, "stats.npz")
    temporary = stats_path + ".tmp"
    with open(temporary, "wb") as stream:
        np.savez(stream, **stats)
    os.replace(temporary, stats_path)
    temporary = manifest_path + ".tmp"
    with open(temporary, "w") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print(
        f"Saved train={len(train)}, val=0, test={len(test)}; statistics use {count} training points only."
    )
    for key, value in stats.items():
        print(key, value.tolist())


if __name__ == "__main__":
    main()
