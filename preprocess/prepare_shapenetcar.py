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
import argparse
import json
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy


FOLDS = tuple(range(9))


def parse_args():
	parser = argparse.ArgumentParser(description="Prepare ShapeNetCar dataset for PANO.")
	parser.add_argument(
		"--input_dir",
		type=str,
		required=True,
		help="ShapeNetCar root directory or its training_data directory.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./dataset/",
		help="Root directory where shapenetcar_fold<id> folders will be created.",
	)
	parser.add_argument(
		"--folds",
		type=int,
		nargs="+",
		default=list(FOLDS),
		choices=FOLDS,
		help="Held-out folds to prepare. Each fold is saved as an independent dataset folder.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing prepared case files.",
	)
	parser.add_argument(
		"--no_extract",
		action="store_true",
		help="Do not auto-extract param*.tar.gz archives when param* directories are missing.",
	)
	parser.add_argument(
		"--visualize",
		action="store_true",
		help="Save simple field visualizations under each output_dir/shapenetcar_fold<id>/vis.",
	)
	parser.add_argument("--n_vis", type=int, default=8)
	parser.add_argument("--vis_points", type=int, default=50000)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def resolve_training_dir(input_dir):
	training_dir = os.path.join(input_dir, "training_data")
	if os.path.isdir(training_dir):
		return training_dir
	return input_dir


def safe_extract(tar, path):
	base_path = os.path.abspath(path)
	for member in tar.getmembers():
		member_path = os.path.abspath(os.path.join(path, member.name))
		if not member_path.startswith(base_path + os.sep) and member_path != base_path:
			raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
	tar.extractall(path)


def ensure_fold_dirs(training_dir, folds, auto_extract):
	for fold_id in folds:
		fold_name = f"param{fold_id}"
		fold_dir = os.path.join(training_dir, fold_name)
		if os.path.isdir(fold_dir):
			continue
		tar_path = os.path.join(training_dir, f"{fold_name}.tar.gz")
		if not os.path.exists(tar_path):
			raise FileNotFoundError(f"Missing {fold_dir} and {tar_path}.")
		if not auto_extract:
			raise FileNotFoundError(f"{fold_dir} is missing. Re-run without --no_extract to extract {tar_path}.")
		print(f"Extracting {tar_path} ...")
		with tarfile.open(tar_path, "r:gz") as tar:
			safe_extract(tar, training_dir)


def list_fold_cases(training_dir):
	fold_cases = {}
	for fold_id in FOLDS:
		fold_name = f"param{fold_id}"
		fold_dir = os.path.join(training_dir, fold_name)
		if not os.path.isdir(fold_dir):
			continue
		case_names = []
		for shape_id in sorted(os.listdir(fold_dir)):
			case_dir = os.path.join(fold_dir, shape_id)
			if not os.path.isdir(case_dir):
				continue
			press_path = os.path.join(case_dir, "quadpress_smpl.vtk")
			velo_path = os.path.join(case_dir, "hexvelo_smpl.vtk")
			if os.path.exists(press_path) and os.path.exists(velo_path):
				case_names.append(f"{fold_name}__{shape_id}")
		fold_cases[fold_id] = case_names
	return fold_cases


def case_id_to_source(case_id):
	fold_name, shape_id = case_id.split("__", 1)
	return fold_name, shape_id, os.path.join(fold_name, shape_id)


def load_unstructured_grid_data(file_name):
	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(file_name)
	reader.Update()
	return reader.GetOutput()


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
	surface_filter = vtk.vtkDataSetSurfaceFilter()
	surface_filter.SetInputData(unstructured_grid_data)
	surface_filter.Update()
	return surface_filter.GetOutput(), surface_filter


def get_sdf(target, boundary):
	tree = cKDTree(boundary)
	dists, indices = tree.query(target, k=1)
	nearest = boundary[indices]
	dirs = (target - nearest) / (dists[:, None] + 1e-8)
	return dists, dirs


def get_normal(unstructured_grid_data):
	poly_data, _ = unstructured_grid_data_to_poly_data(unstructured_grid_data)
	normal_filter = vtk.vtkPolyDataNormals()
	normal_filter.SetInputData(poly_data)
	normal_filter.SetAutoOrientNormals(1)
	normal_filter.SetConsistency(1)
	normal_filter.SetComputeCellNormals(1)
	normal_filter.SetComputePointNormals(0)
	normal_filter.Update()

	unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
	c2p = vtk.vtkCellDataToPointData()
	c2p.SetInputData(unstructured_grid_data)
	c2p.Update()
	normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.float64)
	normal /= np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8
	normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
	if np.isnan(normal).any():
		raise ValueError("NaN values found while computing ShapeNetCar surface normals.")
	return normal


def load_case(training_dir, case_id):
	fold_name, shape_id, source_case = case_id_to_source(case_id)
	case_dir = os.path.join(training_dir, source_case)
	press_path = os.path.join(case_dir, "quadpress_smpl.vtk")
	velo_path = os.path.join(case_dir, "hexvelo_smpl.vtk")
	if not os.path.exists(press_path):
		raise FileNotFoundError(press_path)
	if not os.path.exists(velo_path):
		raise FileNotFoundError(velo_path)

	press_grid = load_unstructured_grid_data(press_path)
	velo_grid = load_unstructured_grid_data(velo_path)

	velo = vtk_to_numpy(velo_grid.GetPointData().GetVectors()).astype(np.float32)
	press = vtk_to_numpy(press_grid.GetPointData().GetScalars()).astype(np.float32)
	points_velo = vtk_to_numpy(velo_grid.GetPoints().GetData()).astype(np.float32)
	points_press = vtk_to_numpy(press_grid.GetPoints().GetData()).astype(np.float32)

	sdf_velo, normal_velo = get_sdf(points_velo, points_press)
	sdf_press = np.zeros(points_press.shape[0], dtype=np.float32)
	normal_press = get_normal(press_grid).astype(np.float32)

	surface_points = {tuple(point) for point in points_press}
	exterior_indices = [idx for idx, point in enumerate(points_velo) if tuple(point) not in surface_points]
	velo_dict = {tuple(point): velo[idx] for idx, point in enumerate(points_velo)}

	pos_ext = points_velo[exterior_indices]
	pos_surf = points_press
	sdf_ext = sdf_velo[exterior_indices].astype(np.float32)
	normal_ext = normal_velo[exterior_indices].astype(np.float32)
	velo_ext = velo[exterior_indices]
	velo_surf = np.array([velo_dict.get(tuple(point), np.zeros(3, dtype=np.float32)) for point in pos_surf], dtype=np.float32)
	press_ext = np.zeros((len(exterior_indices), 1), dtype=np.float32)
	press_surf = press.reshape(-1, 1).astype(np.float32)

	coords = np.concatenate((pos_ext, pos_surf), axis=0).astype(np.float32)
	condition = np.concatenate(
		(
			np.concatenate((pos_ext, sdf_ext[:, None], normal_ext), axis=-1),
			np.concatenate((pos_surf, sdf_press[:, None], normal_press), axis=-1),
		),
		axis=0,
	).astype(np.float32)
	sol = np.concatenate(
		(
			np.concatenate((velo_ext, press_ext), axis=-1),
			np.concatenate((velo_surf, press_surf), axis=-1),
		),
		axis=0,
	).astype(np.float32)
	surf = np.concatenate((np.zeros(pos_ext.shape[0]), np.ones(pos_surf.shape[0]))).astype(np.bool_)

	cd_path = os.path.join(case_dir, "cd.txt")
	cd = np.array(np.nan, dtype=np.float32)
	if os.path.exists(cd_path):
		try:
			cd = np.array(float(np.loadtxt(cd_path)), dtype=np.float32)
		except (TypeError, ValueError):
			cd = np.array(np.nan, dtype=np.float32)

	return {
		"coords": coords,
		"condition": condition,
		"sol": sol,
		"surf": surf,
		"case_name": np.array(case_id),
		"source_case": np.array(source_case),
		"fold": np.array(int(fold_name.replace("param", "")), dtype=np.int64),
		"shape_id": np.array(shape_id),
		"cd": cd,
	}


def build_fold_manifest(fold_cases, fold_id):
	train_cases = []
	for current_fold, case_names in fold_cases.items():
		if current_fold != fold_id:
			train_cases.extend(case_names)
	test_cases = list(fold_cases[fold_id])
	all_cases = sorted(set(train_cases) | set(test_cases))
	return {
		"fold": fold_id,
		"source_train_folds": [fold for fold in sorted(fold_cases) if fold != fold_id],
		"source_test_fold": fold_id,
		"train_full": train_cases,
		"train": train_cases,
		"val": test_cases,
		"test": test_cases,
		"all_cases": all_cases,
		"feature_names": {
			"coords": ["x", "y", "z"],
			"condition": ["x", "y", "z", "sdf", "normal_x", "normal_y", "normal_z"],
			"sol": ["u_x", "u_y", "u_z", "p"],
		},
	}


def compute_stats(prepared_dir, case_names):
	count = 0
	condition_sum = None
	sol_sum = None
	for case_name in tqdm.tqdm(case_names, desc="Stats mean"):
		data = np.load(os.path.join(prepared_dir, "cases", f"{case_name}.npz"))
		condition = data["condition"].astype(np.float64)
		sol = data["sol"].astype(np.float64)
		if condition_sum is None:
			condition_sum = np.zeros(condition.shape[-1], dtype=np.float64)
			sol_sum = np.zeros(sol.shape[-1], dtype=np.float64)
		condition_sum += condition.sum(axis=0)
		sol_sum += sol.sum(axis=0)
		count += condition.shape[0]

	condition_mean = condition_sum / count
	sol_mean = sol_sum / count

	condition_sq_sum = np.zeros_like(condition_mean)
	sol_sq_sum = np.zeros_like(sol_mean)
	for case_name in tqdm.tqdm(case_names, desc="Stats std"):
		data = np.load(os.path.join(prepared_dir, "cases", f"{case_name}.npz"))
		condition = data["condition"].astype(np.float64)
		sol = data["sol"].astype(np.float64)
		condition_sq_sum += ((condition - condition_mean) ** 2).sum(axis=0)
		sol_sq_sum += ((sol - sol_mean) ** 2).sum(axis=0)

	condition_std = np.sqrt(condition_sq_sum / count) + 1e-8
	sol_std = np.sqrt(sol_sq_sum / count) + 1e-8
	return condition_mean.astype(np.float32), condition_std.astype(np.float32), sol_mean.astype(np.float32), sol_std.astype(np.float32)


def select_vis_cases(case_names, n_vis, seed):
	if n_vis == 0:
		return []
	if n_vis < 0 or n_vis >= len(case_names):
		return list(case_names)
	rng = np.random.default_rng(seed)
	indices = np.sort(rng.choice(len(case_names), size=n_vis, replace=False))
	return [case_names[index] for index in indices]


def visualize_cases(prepared_dir, case_names, n_vis, seed, max_points):
	vis_dir = os.path.join(prepared_dir, "vis")
	os.makedirs(vis_dir, exist_ok=True)
	rng = np.random.default_rng(seed)
	for case_name in select_vis_cases(case_names, n_vis, seed):
		data = np.load(os.path.join(prepared_dir, "cases", f"{case_name}.npz"))
		coords = data["coords"]
		sol = data["sol"]
		surf = data["surf"]
		if max_points > 0 and coords.shape[0] > max_points:
			index = rng.choice(coords.shape[0], size=max_points, replace=False)
		else:
			index = np.arange(coords.shape[0])

		fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
		for plot_idx, (field_idx, title) in enumerate([(0, "u_x"), (3, "p"), (3, "surface p")], start=1):
			axis = fig.add_subplot(1, 3, plot_idx, projection="3d")
			if title == "surface p":
				mask = surf
				plot_coords = coords[mask]
				plot_value = sol[mask, field_idx]
			else:
				plot_coords = coords[index]
				plot_value = sol[index, field_idx]
			scatter = axis.scatter(plot_coords[:, 0], plot_coords[:, 1], plot_coords[:, 2], c=plot_value, s=1.0, cmap="coolwarm")
			axis.set_title(f"{title} {case_name}")
			fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
		fig.savefig(os.path.join(vis_dir, f"{case_name}.png"), dpi=150)
		plt.close(fig)


def main():
	args = parse_args()
	training_dir = resolve_training_dir(args.input_dir)
	ensure_fold_dirs(training_dir, FOLDS, auto_extract=not args.no_extract)
	fold_cases = list_fold_cases(training_dir)
	missing_folds = [fold for fold in FOLDS if len(fold_cases.get(fold, [])) == 0]
	if missing_folds:
		raise RuntimeError(f"No ShapeNetCar cases found for folds: {missing_folds}")

	for fold_id in args.folds:
		prepared_dir = os.path.join(args.output_dir, f"shapenetcar_fold{fold_id}")
		cases_dir = os.path.join(prepared_dir, "cases")
		os.makedirs(cases_dir, exist_ok=True)

		prepared_manifest = build_fold_manifest(fold_cases, fold_id)
		for case_name in tqdm.tqdm(prepared_manifest["all_cases"], desc=f"Preparing fold{fold_id} cases"):
			case_path = os.path.join(cases_dir, f"{case_name}.npz")
			if os.path.exists(case_path) and not args.overwrite:
				continue
			case_data = load_case(training_dir, case_name)
			np.savez_compressed(case_path, **case_data)

		with open(os.path.join(prepared_dir, "manifest.json"), "w") as file:
			json.dump(prepared_manifest, file, indent=2)

		condition_mean, condition_std, sol_mean, sol_std = compute_stats(prepared_dir, prepared_manifest["train"])
		np.savez(
			os.path.join(prepared_dir, "stats.npz"),
			condition_mean=condition_mean,
			condition_std=condition_std,
			sol_mean=sol_mean,
			sol_std=sol_std,
		)

		if args.visualize:
			visualize_cases(prepared_dir, prepared_manifest["train"], args.n_vis, args.seed, args.vis_points)

	print("Saved prepared ShapeNetCar datasets:")
	for fold_id in args.folds:
		print(f"  {os.path.join(args.output_dir, f'shapenetcar_fold{fold_id}')}")


if __name__ == "__main__":
	main()
