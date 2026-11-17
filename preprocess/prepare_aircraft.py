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

"""Convert aircraft surface Tecplot files to per-case NPZ files.

Inputs contain coordinates, zero surface SDF, area-weighted vertex normals,
and Mach/alpha/beta. Splits and normalization are prepared separately.
"""

import argparse
import csv
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

VARIABLES = ("x", "y", "z", "Cp", "Rho", "U", "V", "W", "Pressure")
COEFFICIENTS = ("CA", "CN", "CZ", "Cl", "Cn", "Cm")
CONDITION_PATTERN = re.compile(
    r"Mach(?P<mach>[+-]?\d+(?:\.\d+)?)_"
    r"Alpha(?P<alpha>[+-]?\d+(?:\.\d+)?)_"
    r"Beta(?P<beta>[+-]?\d+(?:\.\d+)?)"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, default="raw_datasets/aircraft")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Root directory; the aircraft subdirectory is created inside it.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    return args


def case_identity(path):
    # Raw paths encode geometry and flight conditions as
    # <shape_id>/Mach<value>_Alpha<value>_Beta<value>/Components.i.dat.
    condition_name = os.path.basename(os.path.dirname(path))
    shape_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    match = CONDITION_PATTERN.fullmatch(condition_name)
    shape_id = int(shape_name)
    flight = tuple(float(match[name]) for name in ("mach", "alpha", "beta"))
    name = f"{shape_id}_{flight[0]}_{flight[1]}_{flight[2]}"
    return name, (shape_id, *flight)


def load_coefficients(path):
    # Match CSV records to meshes by geometry ID and all three flight conditions.
    coefficients = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"idx", "Ma", "alpha", "beta", *COEFFICIENTS}

        for row in reader:
            shape_id = float(row["idx"])
            key = (
                int(shape_id),
                *(float(row[name]) for name in ("Ma", "alpha", "beta")),
            )
            values = np.array(
                [float(row[name]) for name in COEFFICIENTS], dtype=np.float64
            )
            coefficients[key] = values
    return coefficients


def read_surface(path):
    """Read the single FEPOINT/TRIANGLE zone without resampling or reordering."""
    with open(path, "rb") as stream:
        title = stream.readline().decode("ascii").strip()
        variables = stream.readline().decode("ascii").strip()
        zone = stream.readline().decode("ascii").strip()
        if not title.startswith("TITLE") or not variables.startswith("VARIABLES"):
            raise ValueError(f"Unexpected Tecplot header: {path}")
        names = tuple(
            v.strip().strip('"') for v in variables.split("=", 1)[1].split(",")
        )
        if names != VARIABLES:
            raise ValueError(f"Unexpected variables in {path}: {names}")

        # Validate the zone layout before interpreting the numeric payload.
        point_match = re.search(r"\bN\s*=\s*(\d+)", zone)
        face_match = re.search(r"\bE\s*=\s*(\d+)", zone)
        n_point, n_face = int(point_match[1]), int(face_match[1])

        # Bulk parsing is substantially faster than line-by-line Python parsing.
        values = np.fromstring(stream.read().decode("ascii"), sep=" ", dtype=np.float64)

    # FEPOINT stores all point fields first, followed by triangle connectivity.
    expected = n_point * len(VARIABLES) + n_face * 3
    fields = values[: n_point * len(VARIABLES)].reshape(n_point, len(VARIABLES))
    connectivity = values[n_point * len(VARIABLES) :].reshape(n_face, 3)

    # Convert Tecplot's 1-based indices to NumPy's 0-based indexing.
    faces = connectivity.astype(np.int32) - 1
    return fields, faces


def vertex_normals(coords, faces):
    """Area-weighted unit normals; preserve the source mesh's orientation."""
    triangles = coords[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)

    # Each cross product has magnitude twice the triangle area. Summing these
    # vectors at incident vertices provides the area weighting before normalization.
    normals = np.zeros_like(coords, dtype=np.float64)
    for corner in range(3):
        for axis in range(3):
            normals[:, axis] += np.bincount(
                faces[:, corner], weights=cross[:, axis], minlength=len(coords)
            )

    lengths = np.linalg.norm(normals, axis=1)
    normals /= lengths[:, None]

    # Signed volume is recorded as a diagnostic, never used to flip the mesh.
    volume = np.einsum("ij,ij->i", triangles[:, 0], cross).sum() / 6
    return normals.astype(np.float32), {
        "surface_area": float(double_area.sum() / 2),
        "signed_volume": float(volume),
        "normal_length_max_error": float(
            np.abs(np.linalg.norm(normals, axis=1) - 1).max()
        ),
    }


def prepare_case(job):
    source, output, aero = job
    name, (shape_id, mach, alpha, beta) = case_identity(source)
    fields, faces = read_surface(source)
    normals, geometry = vertex_normals(fields[:, :3], faces)

    # Keep the original point order and six solution fields. Every input point
    # lies on the surface, so SDF is zero and flight conditions are broadcast.
    coords = fields[:, :3].astype(np.float32)
    sol = fields[:, 3:].astype(np.float32)
    flight = np.array([mach, alpha, beta], dtype=np.float32)
    condition = np.concatenate(
        (
            coords,
            np.zeros((len(coords), 1), dtype=np.float32),
            normals,
            np.broadcast_to(flight, (len(coords), 3)),
        ),
        axis=1,
    )

    data = {
        "coords": coords,
        "condition": condition,
        "sol": sol,
        "surf": np.ones(len(coords), dtype=np.bool_),
        "faces": faces,
        "case_name": np.array(name),
        "source_case": np.array(
            os.path.join(str(shape_id), os.path.basename(os.path.dirname(source)))
        ),
        "shape_id": np.array(shape_id, dtype=np.int64),
        "mach": np.array(mach, dtype=np.float32),
        "alpha_deg": np.array(alpha, dtype=np.float32),
        "beta_deg": np.array(beta, dtype=np.float32),
        "aero_coefficients": aero,
    }

    # Avoid leaving a seemingly valid case file after an interrupted write.
    temporary = os.path.splitext(output)[0] + ".npz.tmp"
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, output)
    return {
        "case_name": name,
        "source_case": str(data["source_case"]),
        "shape_id": shape_id,
        "mach": mach,
        "alpha_deg": alpha,
        "beta_deg": beta,
        "n_points": len(coords),
        "n_faces": len(faces),
        "coords_min": coords.min(axis=0).tolist(),
        "coords_max": coords.max(axis=0).tolist(),
        "sol_min": sol.min(axis=0).tolist(),
        "sol_max": sol.max(axis=0).tolist(),
        **geometry,
    }


def main():
    args = parse_args()

    # Numeric geometry/flight ordering keeps case and metadata order reproducible.
    sources = sorted(
        glob.glob(
            os.path.join(glob.escape(args.input_dir), "*", "*", "Components.i.dat")
        ),
        key=lambda p: case_identity(p)[1],
    )
    if not sources:
        raise FileNotFoundError(f"No Components.i.dat files under {args.input_dir}")
    names = [case_identity(path)[0] for path in sources]

    coefficients = load_coefficients(os.path.join(args.input_dir, "result.csv"))
    keys = [case_identity(path)[1] for path in sources]

    prepared_dir = os.path.join(args.output_dir, "aircraft")
    cases_dir = os.path.join(prepared_dir, "cases")
    manifest_path = os.path.join(prepared_dir, "manifest.json")

    # Never silently destroy a later user-supplied split or leave stale stats.
    if os.path.exists(manifest_path):
        with open(manifest_path) as stream:
            previous = json.load(stream)
    existing_cases = glob.glob(os.path.join(glob.escape(cases_dir), "*.npz"))
    if not args.overwrite and existing_cases:
        raise FileExistsError(
            "Prepared cases already exist; pass --overwrite to regenerate them."
        )
    existing_names = {
        os.path.splitext(os.path.basename(path))[0] for path in existing_cases
    }
    os.makedirs(cases_dir, exist_ok=True)

    # Convert each complete mesh independently; both map implementations preserve
    # job order so the manifest remains aligned with the sorted source cases.
    jobs = [
        (p, os.path.join(cases_dir, f"{name}.npz"), coefficients[key])
        for p, name, key in zip(sources, names, keys)
    ]
    if args.workers == 1:
        metadata = list(
            tqdm(map(prepare_case, jobs), total=len(jobs), desc="Preparing aircraft")
        )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            metadata = list(
                tqdm(
                    pool.map(prepare_case, jobs),
                    total=len(jobs),
                    desc="Preparing aircraft",
                )
            )
    # Split assignment and training-only statistics are handled separately by
    # prepare_aircraft_split.py; mark both as pending in this conversion manifest.
    manifest = {
        "task": "aircraft",
        "format_version": 1,
        "split_status": "pending",
        "statistics_status": "pending_training_split",
        "train_full": [],
        "train": [],
        "val": [],
        "test": [],
        "all_cases": names,
        "feature_names": {
            "coords": list(VARIABLES[:3]),
            "condition": [
                "x",
                "y",
                "z",
                "sdf",
                "normal_x",
                "normal_y",
                "normal_z",
                "mach",
                "alpha_deg",
                "beta_deg",
            ],
            "sol": list(VARIABLES[3:]),
            "aero_coefficients": list(COEFFICIENTS),
        },
        "source": os.path.realpath(args.input_dir),
        "case_metadata": metadata,
    }
    temporary = manifest_path + ".tmp"
    with open(temporary, "w") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print(
        f"Saved {len(names)} aircraft cases to {prepared_dir}. No split or statistics generated."
    )


if __name__ == "__main__":
    main()
