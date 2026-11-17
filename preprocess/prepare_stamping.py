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
"""Convert official Unilateral Stamping ZIP to per-case NPZ, preserving splits/cells."""

import argparse
import gc
import json
import os
import pickle
import zipfile

import numpy as np
from tqdm import tqdm

BODY_NAMES = ["metal", "rubber", "fixed_die"] + [
    f"moving_die_{i}" for i in range(1, 17)
]


def convert_case(sample):
    # Concatenate bodies in a fixed order: metal, rubber, fixed die, then the
    # 16 moving dies. Offsets retain each body's slice in the combined cloud.
    keys = ["init_metal_pos", "init_rubber_pos", "die_1_pos"] + [
        f"die_2-{i}_pos" for i in range(1, 17)
    ]
    arrays = [np.asarray(sample[k]) for k in keys]
    offsets = np.concatenate(([0], np.cumsum([len(a) for a in arrays])))
    coords = np.concatenate([a[:, :3] for a in arrays]).astype(np.float32)
    body_id = np.repeat(np.arange(19), np.diff(offsets)).astype(np.int64)

    # Moving-die inputs contain xyz followed by prescribed displacement.
    # Other bodies receive zero prescribed displacement.
    load = np.zeros_like(coords)
    for i in range(3, 19):
        assert arrays[i].shape[1] == 6
        load[offsets[i] : offsets[i + 1]] = arrays[i][:, 3:]

    # Only metal and rubber have prediction targets. Store displacement from
    # the initial geometry, stress, and equivalent plastic strain (PEEQ).
    sol = np.zeros((len(coords), 5), dtype=np.float32)
    for i, material in enumerate(["metal", "rubber"]):
        initial = np.asarray(sample[f"init_{material}_pos"])
        final = np.asarray(sample[f"final_{material}_pos"])
        assert initial.shape == final.shape
        sol[offsets[i] : offsets[i + 1]] = np.concatenate(
            [
                final - initial,
                np.asarray(sample[f"final_{material}_stress"]).reshape(-1, 1),
                np.asarray(sample[f"final_{material}_peeq"]).reshape(-1, 1),
            ],
            axis=1,
        )

    result = dict(
        coords=coords,
        prescribed_displacement=load,
        body_id=body_id,
        body_offsets=offsets,
        target_mask=body_id < 2,
        load_mask=body_id >= 3,
        sol=sol,
    )

    # Preserve official 1-based connectivity and -1 triangle padding without conversion.
    result.update({k: np.asarray(v) for k, v in sample.items() if k.endswith("_cells")})
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite sample values.")
    return result


def prepare(source, output):
    os.makedirs(output, exist_ok=True)
    manifest_path = os.path.join(output, "manifest.json")
    cases_dir = os.path.join(output, "cases")
    if os.path.exists(manifest_path):
        raise FileExistsError(f"{manifest_path} already exists; use a new destination.")
    os.makedirs(cases_dir, exist_ok=True)

    manifest = dict(
        format_version=1,
        body_names=BODY_NAMES,
        input_channels=25,
        output_channels=["ux", "uy", "uz", "stress", "peeq"],
        cells_index_base=1,
        cells_padding=-1,
        source=str(source),
        train=[],
        val=[],
        test=[],
    )

    # Accumulate training moments for shared geometry normalization, moving-die
    # loads, and separate metal/rubber target statistics.
    coord_sum = np.zeros(3)
    coord_sq = np.zeros(3)
    count = 0

    load_sq = 0.0
    load_count = 0

    sums = np.zeros((2, 5))
    squares = np.zeros((2, 5))
    counts = np.zeros(2)

    with zipfile.ZipFile(source) as archive:
        # Preserve the official split membership; rename "eval" to "val" only.
        for original, split in [("train", "train"), ("eval", "val"), ("test", "test")]:
            member = f"Unisoma_Dataset/Unilateral_Stamping/unilateral_stamping_dataset_{original}.pkl"
            with archive.open(member) as stream:
                raw = pickle.load(stream)

            items = raw.items() if isinstance(raw, dict) else enumerate(raw)
            for index, (source_id, sample) in enumerate(
                tqdm(items, total=len(raw), desc=split)
            ):
                case = convert_case(sample)
                case["source_id"] = np.asarray(str(source_id))
                name = f"{split}_{index:06d}"
                np.savez_compressed(os.path.join(cases_dir, f"{name}.npz"), **case)
                manifest[split].append(name)

                # Validation and test cases must not contribute to normalization.
                if split == "train":
                    x = case["coords"].astype(np.float64)
                    coord_sum += x.sum(0)
                    coord_sq += (x * x).sum(0)
                    count += len(x)

                    # Count each moving die once, regardless of its mesh size.
                    # All points belonging to that die share the same translation.
                    for i in range(1, 17):
                        d = np.asarray(sample[f"die_2-{i}_pos"])[:, 3:]
                        load_sq += np.square(d[0]).sum()
                        load_count += 1

                    # Each material has its own mean/std for all five targets.
                    for i in range(2):
                        y = case["sol"][case["body_id"] == i].astype(np.float64)
                        sums[i] += y.sum(0)
                        squares[i] += (y * y).sum(0)
                        counts[i] += len(y)

            # Release the complete split before loading the next large pickle.
            del raw, items, sample
            gc.collect()

    # Geometry uses a shared center and scalar RMS radius; loads use an
    # uncentered RMS magnitude. Targets use per-material population statistics.
    mean = coord_sum / count
    scale = np.sqrt(np.maximum(coord_sq / count - mean**2, 0).sum())

    ymean = sums / counts[:, None]
    ystd = np.sqrt(np.maximum(squares / counts[:, None] - ymean**2, 0))

    # Guard constant channels against division by zero in downstream normalization.
    np.savez(
        os.path.join(output, "stats.npz"),
        coords_mean=mean,
        coords_scale=max(scale, 1e-8),
        load_scale=max(np.sqrt(load_sq / load_count), 1e-8),
        sol_mean=ymean,
        sol_std=np.maximum(ystd, 1e-8),
    )

    manifest["normalization"] = dict(
        statistics_split="train",
        coords="shared center and scalar RMS radius",
        load="uncentered RMS vector magnitude; one vote per moving die",
        target="per material, per channel population mean/std",
    )
    with open(manifest_path, "w") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="raw_datasets/Unisoma/Unisoma_Dataset.zip")
    parser.add_argument("--output", default="dataset/unilateral_stamping")
    args = parser.parse_args()
    prepare(args.source, args.output)
