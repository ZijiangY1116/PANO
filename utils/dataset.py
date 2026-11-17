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

import json
import os

import h5py
import numpy as np
import torch
import tqdm


class StaticDataset(torch.utils.data.Dataset):

    """
    This is the basic dataset.
    Support:
    Input: coords, condition (e.g. coeff), output: sol

    Darcy:
    - coords: (N, W, H, 2) grid coordinates
    - condition: (N, W, H, 1) coefficient field
    - sol: (N, W, H, 1) solution field
    """

    class Normalizer:
        def __init__(self, x, y1, y2):
            self.x_flag = False
            self.y1_flag = False
            self.y2_flag = False

            old_x_shape = x.shape
            old_y1_shape = y1.shape
            old_y2_shape = y2.shape

            x = torch.reshape(x, (-1, x.shape[-1]))
            y1 = torch.reshape(y1, (-1, y1.shape[-1]))
            y2 = torch.reshape(y2, (-1, y2.shape[-1]))

            self.x_mean = torch.mean(x, dim=0)
            self.x_std = torch.std(x, dim=0) + 1e-8
            self.y1_mean = torch.mean(y1, dim=0)
            self.y1_std = torch.std(y1, dim=0) + 1e-8
            self.y2_mean = torch.mean(y2, dim=0)
            self.y2_std = torch.std(y2, dim=0) + 1e-8

            x = torch.reshape(x, old_x_shape)
            y1 = torch.reshape(y1, old_y1_shape)
            y2 = torch.reshape(y2, old_y2_shape)

        def set(self, x_mean=None, x_std=None, y1_mean=None, y1_std=None, y2_mean=None, y2_std=None):
            if x_mean is not None:
                self.x_mean = x_mean.detach().clone()
            if x_std is not None:
                self.x_std = x_std.detach().clone()
            if y1_mean is not None:
                self.y1_mean = y1_mean.detach().clone()
            if y1_std is not None:
                self.y1_std = y1_std.detach().clone()
            if y2_mean is not None:
                self.y2_mean = y2_mean.detach().clone()
            if y2_std is not None:
                self.y2_std = y2_std.detach().clone()

        def export(self):
            return {
                "x_mean": self.x_mean.detach().clone(),
                "x_std": self.x_std.detach().clone(),
                "y1_mean": self.y1_mean.detach().clone(),
                "y1_std": self.y1_std.detach().clone(),
                "y2_mean": self.y2_mean.detach().clone(),
                "y2_std": self.y2_std.detach().clone(),
            }

        def is_apply_x(self):
            return self.x_flag

        def is_apply_y1(self):
            return self.y1_flag

        def is_apply_y2(self):
            return self.y2_flag

        def apply_x(self, x, inverse=False):
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)

            old_x_shape = x.shape
            x = torch.reshape(x, (-1, x.shape[-1]))
            if not inverse:
                x = (x - self.x_mean) / self.x_std
                self.x_flag = True
            else:
                x = x * self.x_std + self.x_mean
            return torch.reshape(x, old_x_shape)

        def apply_y1(self, y1, inverse=False):
            self.y1_mean = self.y1_mean.to(y1.device)
            self.y1_std = self.y1_std.to(y1.device)

            old_y1_shape = y1.shape
            y1 = torch.reshape(y1, (-1, y1.shape[-1]))
            if not inverse:
                y1 = (y1 - self.y1_mean) / self.y1_std
                self.y1_flag = True
            else:
                y1 = y1 * self.y1_std + self.y1_mean
            return torch.reshape(y1, old_y1_shape)

        def apply_y2(self, y2, inverse=False):
            self.y2_mean = self.y2_mean.to(y2.device)
            self.y2_std = self.y2_std.to(y2.device)

            old_y2_shape = y2.shape
            y2 = torch.reshape(y2, (-1, y2.shape[-1]))
            if not inverse:
                y2 = (y2 - self.y2_mean) / self.y2_std
                self.y2_flag = True
            else:
                y2 = y2 * self.y2_std + self.y2_mean
            return torch.reshape(y2, old_y2_shape)

    def __init__(self, data_mode, apply_norm=True, coords_condition=True, data_root="./dataset", norm_params=None):
        super().__init__()
        data_file = os.path.join(data_root, f"{data_mode}.npy")
        dataset = np.load(data_file, allow_pickle=True).tolist()

        self.coords = torch.tensor(dataset["coords"]).float()
        if coords_condition:
            self.condition = torch.cat((self.coords, torch.tensor(dataset["condition"]).float()), dim=-1)
        else:
            self.condition = torch.tensor(dataset["condition"]).float()

        self.sol = torch.tensor(dataset["sol"]).float()

        self.normalizer = StaticDataset.Normalizer(self.coords, self.condition, self.sol)
        if norm_params is not None:
            self.normalizer.set(**norm_params)
        if apply_norm:
            self.coords = self.normalizer.apply_x(self.coords)
            self.condition = self.normalizer.apply_y1(self.condition)
            self.sol = self.normalizer.apply_y2(self.sol)

        self.length = self.coords.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.condition[idx], self.sol[idx]

    def dim(self):
        return self.coords.shape[-1], self.condition.shape[-1], self.sol.shape[-1]

    def get_normalizer(self):
        return self.normalizer


class AirfRANSDataset(torch.utils.data.Dataset):

    """
    AirfRANS point-cloud dataset prepared by preprocess/prepare_airfrans.py.

    Each prepared case stores variable-length full mesh points:
    - coords: (P, 2), physical x/y coordinates
    - condition: (P, 7), [x, y, u_inf_x, u_inf_y, sdf, normal_x, normal_y]
    - sol: (P, 4), [u_x, u_y, p, nut]
    - surf: (P,), boolean surface mask

    When subsampling is set, __getitem__ returns a fixed number of points and
    is suitable for ordinary dense batching.
    """

    class Normalizer:
        def __init__(self, condition_mean, condition_std, sol_mean, sol_std):
            self.x_flag = False
            self.y1_flag = False
            self.y2_flag = False
            self.y1_mean = torch.as_tensor(condition_mean, dtype=torch.float32).detach().clone()
            self.y1_std = torch.as_tensor(condition_std, dtype=torch.float32).detach().clone()
            self.x_mean = self.y1_mean[:2].detach().clone()
            self.x_std = self.y1_std[:2].detach().clone()
            self.y2_mean = torch.as_tensor(sol_mean, dtype=torch.float32).detach().clone()
            self.y2_std = torch.as_tensor(sol_std, dtype=torch.float32).detach().clone()

        def set(self, x_mean=None, x_std=None, y1_mean=None, y1_std=None, y2_mean=None, y2_std=None):
            if x_mean is not None:
                self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32).detach().clone()
            if x_std is not None:
                self.x_std = torch.as_tensor(x_std, dtype=torch.float32).detach().clone()
            if y1_mean is not None:
                self.y1_mean = torch.as_tensor(y1_mean, dtype=torch.float32).detach().clone()
            if y1_std is not None:
                self.y1_std = torch.as_tensor(y1_std, dtype=torch.float32).detach().clone()
            if y2_mean is not None:
                self.y2_mean = torch.as_tensor(y2_mean, dtype=torch.float32).detach().clone()
            if y2_std is not None:
                self.y2_std = torch.as_tensor(y2_std, dtype=torch.float32).detach().clone()

        def export(self):
            return {
                "x_mean": self.x_mean.detach().clone(),
                "x_std": self.x_std.detach().clone(),
                "y1_mean": self.y1_mean.detach().clone(),
                "y1_std": self.y1_std.detach().clone(),
                "y2_mean": self.y2_mean.detach().clone(),
                "y2_std": self.y2_std.detach().clone(),
            }

        def is_apply_x(self):
            return self.x_flag

        def is_apply_y1(self):
            return self.y1_flag

        def is_apply_y2(self):
            return self.y2_flag

        def apply_x(self, x, inverse=False):
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
            if inverse:
                return x * self.x_std + self.x_mean
            self.x_flag = True
            return (x - self.x_mean) / self.x_std

        def apply_y1(self, y1, inverse=False):
            self.y1_mean = self.y1_mean.to(y1.device)
            self.y1_std = self.y1_std.to(y1.device)
            if inverse:
                return y1 * self.y1_std + self.y1_mean
            self.y1_flag = True
            return (y1 - self.y1_mean) / self.y1_std

        def apply_y2(self, y2, inverse=False):
            self.y2_mean = self.y2_mean.to(y2.device)
            self.y2_std = self.y2_std.to(y2.device)
            if inverse:
                return y2 * self.y2_std + self.y2_mean
            self.y2_flag = True
            return (y2 - self.y2_mean) / self.y2_std

    def __init__(
        self,
        data_mode,
        apply_norm=True,
        data_root="./dataset/airfrans_full",
        norm_params=None,
        subsampling=None,
        sampling_mode="random",
        sampling_seed=None,
        include_metadata=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.data_mode = data_mode
        self.apply_norm = apply_norm
        self.subsampling = subsampling
        self.sampling_mode = sampling_mode
        self.include_metadata = include_metadata
        self.rng = np.random.default_rng(0 if sampling_seed is None else int(sampling_seed))

        manifest_path = os.path.join(data_root, "manifest.json")
        stats_path = os.path.join(data_root, "stats.npz")
        with open(manifest_path, "r") as file:
            self.manifest = json.load(file)

        split_key = data_mode
        if split_key not in self.manifest:
            raise KeyError(f"Split {split_key} is not found in {manifest_path}.")
        self.case_names = list(self.manifest[split_key])

        if norm_params is not None:
            self.normalizer = AirfRANSDataset.Normalizer(
                condition_mean=norm_params["y1_mean"],
                condition_std=norm_params["y1_std"],
                sol_mean=norm_params["y2_mean"],
                sol_std=norm_params["y2_std"],
            )
            self.normalizer.set(**norm_params)
        else:
            stats = np.load(stats_path)
            self.normalizer = AirfRANSDataset.Normalizer(
                condition_mean=stats["condition_mean"],
                condition_std=stats["condition_std"],
                sol_mean=stats["sol_mean"],
                sol_std=stats["sol_std"],
            )

        if sampling_mode not in ("random", "first", "all", "all_surface"):
            raise ValueError(f"Unknown sampling_mode {sampling_mode}.")
        if sampling_mode == "all" and subsampling is not None:
            raise ValueError("sampling_mode='all' requires subsampling=None.")
        if sampling_mode == "all_surface" and subsampling is None:
            raise ValueError("sampling_mode='all_surface' requires a positive subsampling value.")
        if subsampling is not None and subsampling <= 0:
            raise ValueError("subsampling must be positive when set.")

    def __len__(self):
        return len(self.case_names)

    def _load_case(self, case_name):
        case_path = os.path.join(self.data_root, "cases", f"{case_name}.npz")
        if not os.path.exists(case_path):
            raise FileNotFoundError(case_path)
        return np.load(case_path)

    def _sample_indices(self, n_point, surf=None):
        if self.subsampling is None or self.sampling_mode == "all":
            return np.arange(n_point)
        if self.sampling_mode == "first":
            if n_point >= self.subsampling:
                return np.arange(self.subsampling)
            return np.resize(np.arange(n_point), self.subsampling)
        if self.sampling_mode == "all_surface":
            if surf is None:
                raise ValueError("surf mask is required for sampling_mode='all_surface'.")
            surf_indices = np.flatnonzero(surf.astype(np.bool_))
            volume_indices = np.flatnonzero(~surf.astype(np.bool_))
            n_volume = self.subsampling - surf_indices.shape[0]
            if n_volume <= 0:
                raise ValueError(
                    f"Expected fewer surface points than subsampling={self.subsampling}, "
                    f"got {surf_indices.shape[0]} surface points."
                )
            volume_sample = self.rng.choice(volume_indices, size=n_volume, replace=False)
            indices = np.concatenate((surf_indices, volume_sample))
            self.rng.shuffle(indices)
            return indices
        replace = n_point < self.subsampling
        return self.rng.choice(n_point, size=self.subsampling, replace=replace)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        data = self._load_case(case_name)
        indices = self._sample_indices(data["coords"].shape[0], data["surf"])

        coords = torch.from_numpy(data["coords"][indices]).float()
        condition = torch.from_numpy(data["condition"][indices]).float()
        sol = torch.from_numpy(data["sol"][indices]).float()
        surf = torch.from_numpy(data["surf"][indices].astype(np.bool_))

        if self.apply_norm:
            coords = self.normalizer.apply_x(coords)
            condition = self.normalizer.apply_y1(condition)
            sol = self.normalizer.apply_y2(sol)

        if self.include_metadata:
            return coords, condition, sol, surf, case_name
        return coords, condition, sol, surf

    def dim(self):
        return 2, 7, 4

    def get_normalizer(self):
        return self.normalizer

    def get_case_names(self):
        return list(self.case_names)




class ShapeNetCarDataset(torch.utils.data.Dataset):

    """
    ShapeNetCar point-cloud dataset prepared by preprocess/prepare_shapenetcar.py.

    Each prepared case stores:
    - coords: (P, 3), physical x/y/z coordinates
    - condition: (P, 7), [x, y, z, sdf, normal_x, normal_y, normal_z]
    - sol: (P, 4), [u_x, u_y, u_z, p]
    - surf: (P,), boolean surface mask
    """

    class Normalizer:
        def __init__(self, condition_mean, condition_std, sol_mean, sol_std):
            self.x_flag = False
            self.y1_flag = False
            self.y2_flag = False
            self.y1_mean = torch.as_tensor(condition_mean, dtype=torch.float32).detach().clone()
            self.y1_std = torch.as_tensor(condition_std, dtype=torch.float32).detach().clone()
            self.x_mean = self.y1_mean[:3].detach().clone()
            self.x_std = self.y1_std[:3].detach().clone()
            self.y2_mean = torch.as_tensor(sol_mean, dtype=torch.float32).detach().clone()
            self.y2_std = torch.as_tensor(sol_std, dtype=torch.float32).detach().clone()

        def set(self, x_mean=None, x_std=None, y1_mean=None, y1_std=None, y2_mean=None, y2_std=None):
            if x_mean is not None:
                self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32).detach().clone()
            if x_std is not None:
                self.x_std = torch.as_tensor(x_std, dtype=torch.float32).detach().clone()
            if y1_mean is not None:
                self.y1_mean = torch.as_tensor(y1_mean, dtype=torch.float32).detach().clone()
            if y1_std is not None:
                self.y1_std = torch.as_tensor(y1_std, dtype=torch.float32).detach().clone()
            if y2_mean is not None:
                self.y2_mean = torch.as_tensor(y2_mean, dtype=torch.float32).detach().clone()
            if y2_std is not None:
                self.y2_std = torch.as_tensor(y2_std, dtype=torch.float32).detach().clone()

        def export(self):
            return {
                "x_mean": self.x_mean.detach().clone(),
                "x_std": self.x_std.detach().clone(),
                "y1_mean": self.y1_mean.detach().clone(),
                "y1_std": self.y1_std.detach().clone(),
                "y2_mean": self.y2_mean.detach().clone(),
                "y2_std": self.y2_std.detach().clone(),
            }

        def is_apply_x(self):
            return self.x_flag

        def is_apply_y1(self):
            return self.y1_flag

        def is_apply_y2(self):
            return self.y2_flag

        def apply_x(self, x, inverse=False):
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
            if inverse:
                return x * self.x_std + self.x_mean
            self.x_flag = True
            return (x - self.x_mean) / self.x_std

        def apply_y1(self, y1, inverse=False):
            self.y1_mean = self.y1_mean.to(y1.device)
            self.y1_std = self.y1_std.to(y1.device)
            if inverse:
                return y1 * self.y1_std + self.y1_mean
            self.y1_flag = True
            return (y1 - self.y1_mean) / self.y1_std

        def apply_y2(self, y2, inverse=False):
            self.y2_mean = self.y2_mean.to(y2.device)
            self.y2_std = self.y2_std.to(y2.device)
            if inverse:
                return y2 * self.y2_std + self.y2_mean
            self.y2_flag = True
            return (y2 - self.y2_mean) / self.y2_std

    def __init__(
        self,
        data_mode,
        apply_norm=True,
        data_root="./dataset/shapenetcar_fold0",
        norm_params=None,
        subsampling=None,
        sampling_mode="random",
        sampling_seed=None,
        include_metadata=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.data_mode = data_mode
        self.apply_norm = apply_norm
        self.subsampling = subsampling
        self.sampling_mode = sampling_mode
        self.include_metadata = include_metadata
        self.rng = np.random.default_rng(0 if sampling_seed is None else int(sampling_seed))

        manifest_path = os.path.join(data_root, "manifest.json")
        stats_path = os.path.join(data_root, "stats.npz")
        with open(manifest_path, "r") as file:
            self.manifest = json.load(file)

        if data_mode not in self.manifest:
            raise KeyError(f"Split {data_mode} is not found in {manifest_path}.")
        self.case_names = list(self.manifest[data_mode])

        if norm_params is not None:
            self.normalizer = ShapeNetCarDataset.Normalizer(
                condition_mean=norm_params["y1_mean"],
                condition_std=norm_params["y1_std"],
                sol_mean=norm_params["y2_mean"],
                sol_std=norm_params["y2_std"],
            )
            self.normalizer.set(**norm_params)
        else:
            stats = np.load(stats_path)
            self.normalizer = ShapeNetCarDataset.Normalizer(
                condition_mean=stats["condition_mean"],
                condition_std=stats["condition_std"],
                sol_mean=stats["sol_mean"],
                sol_std=stats["sol_std"],
            )

        if sampling_mode not in ("random", "first", "all"):
            raise ValueError(f"Unknown sampling_mode {sampling_mode}.")

        if subsampling == -1:
            # defult to all points if subsampling is -1
            self.subsampling = None

    def __len__(self):
        return len(self.case_names)

    def _load_case(self, case_name):
        case_path = os.path.join(self.data_root, "cases", f"{case_name}.npz")
        if not os.path.exists(case_path):
            raise FileNotFoundError(case_path)
        return np.load(case_path)

    def _sample_indices(self, n_point):
        if self.subsampling is None or self.sampling_mode == "all":
            return np.arange(n_point)
        if self.sampling_mode == "first":
            if n_point >= self.subsampling:
                return np.arange(self.subsampling)
            return np.resize(np.arange(n_point), self.subsampling)
        replace = n_point < self.subsampling
        return self.rng.choice(n_point, size=self.subsampling, replace=replace)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        data = self._load_case(case_name)
        indices = self._sample_indices(data["coords"].shape[0])

        coords = torch.from_numpy(data["coords"][indices]).float()
        condition = torch.from_numpy(data["condition"][indices]).float()
        sol = torch.from_numpy(data["sol"][indices]).float()
        surf = torch.from_numpy(data["surf"][indices].astype(np.bool_))

        if self.apply_norm:
            coords = self.normalizer.apply_x(coords)
            condition = self.normalizer.apply_y1(condition)
            sol = self.normalizer.apply_y2(sol)

        if self.include_metadata:
            return coords, condition, sol, surf, case_name
        return coords, condition, sol, surf

    def dim(self):
        return 3, 7, 4

    def get_normalizer(self):
        return self.normalizer

    def get_case_names(self):
        return list(self.case_names)


class AircraftDataset(torch.utils.data.Dataset):
    """Lazy per-case Aircraft data using the static (coords, condition, sol) API."""

    class Normalizer(StaticDataset.Normalizer):
        def __init__(self, condition_mean, condition_std, sol_mean, sol_std):
            self.x_flag = self.y1_flag = self.y2_flag = False
            mean = torch.as_tensor(condition_mean, dtype=torch.float32)
            std = torch.as_tensor(condition_std, dtype=torch.float32)
            self.set(x_mean=mean[:3], x_std=std[:3], y1_mean=mean, y1_std=std,
                     y2_mean=torch.as_tensor(sol_mean, dtype=torch.float32),
                     y2_std=torch.as_tensor(sol_std, dtype=torch.float32))

    def __init__(self, data_mode, apply_norm=True, coords_condition=False,
                 data_root="./dataset/aircraft", norm_params=None):
        super().__init__()
        if coords_condition:
            raise ValueError("Aircraft condition already contains xyz; set coords_condition=False.")
        self.data_root = data_root
        self.apply_norm = apply_norm
        with open(os.path.join(data_root, "manifest.json")) as stream:
            self.manifest = json.load(stream)
        self.case_names = list(self.manifest[data_mode])
        if not self.case_names:
            raise ValueError(f"Aircraft split {data_mode!r} is empty.")
        with np.load(os.path.join(data_root, "stats.npz"), allow_pickle=False) as stats:
            self.normalizer = self.Normalizer(
                stats["condition_mean"], stats["condition_std"],
                stats["sol_mean"], stats["sol_std"],
            )
        if norm_params is not None:
            self.normalizer.set(**norm_params)
        # With worker processes, __getitem__ changes only each worker's copy.
        # The parent normalizer must know the dataset's output scale upfront.
        self.normalizer.x_flag = self.normalizer.y1_flag = self.normalizer.y2_flag = bool(apply_norm)

    def __len__(self):
        return len(self.case_names)

    def case_path(self, index):
        return os.path.join(self.data_root, "cases", self.case_names[index] + ".npz")

    def load_raw_case(self, index):
        with np.load(self.case_path(index), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def __getitem__(self, index):
        with np.load(self.case_path(index), allow_pickle=False) as data:
            coords = torch.from_numpy(data["coords"]).float()
            condition = torch.from_numpy(data["condition"]).float()
            sol = torch.from_numpy(data["sol"]).float()
        if self.apply_norm:
            coords = self.normalizer.apply_x(coords)
            condition = self.normalizer.apply_y1(condition)
            sol = self.normalizer.apply_y2(sol)
        return coords, condition, sol

    def dim(self):
        return 3, 10, 6

    def get_normalizer(self):
        return self.normalizer

    def get_case_names(self):
        return list(self.case_names)


class StampingDataset(torch.utils.data.Dataset):
    """One coupled point cloud; only body IDs 0 (metal) and 1 (rubber) are supervised."""

    class Normalizer:
        def __init__(self, params, apply_norm=True):
            self.params = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in params.items()}
            self.enabled = apply_norm

        def export(self):
            return {k: v.detach().clone() for k, v in self.params.items()}

        def is_apply_y2(self):
            return self.enabled

        def apply_x(self, x, inverse=False):
            if not self.enabled:
                return x
            mean = self.params['coords_mean'].to(x)
            scale = self.params['coords_scale'].to(x)
            return x * scale + mean if inverse else (x - mean) / scale

        def apply_load(self, load):
            return load / self.params['load_scale'].to(load) if self.enabled else load

        def apply_y2(self, y, body_id, inverse=False):
            if not self.enabled:
                return y
            valid = body_id < 2
            material = body_id.clamp(0, 1)
            mean = self.params['sol_mean'].to(y)[material]
            std = self.params['sol_std'].to(y)[material]
            transformed = y * std + mean if inverse else (y - mean) / std
            return torch.where(valid.unsqueeze(-1), transformed, torch.zeros_like(y))

    def __init__(self, data_mode, apply_norm=True, data_root='./dataset/unilateral_stamping', norm_params=None):
        self.data_root = data_root
        with open(os.path.join(data_root, 'manifest.json')) as stream:
            self.manifest = json.load(stream)
        self.case_names = list(self.manifest['val' if data_mode == 'eval' else data_mode])
        if not self.case_names:
            raise ValueError(f'Empty Stamping split: {data_mode}')
        if norm_params is None:
            with np.load(os.path.join(data_root, 'stats.npz')) as stats:
                norm_params = {k: stats[k] for k in stats.files}
        self.normalizer = self.Normalizer(norm_params, apply_norm)

    def __len__(self):
        return len(self.case_names)

    def load_raw_case(self, index):
        path = os.path.join(self.data_root, 'cases', self.case_names[index] + '.npz')
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def __getitem__(self, index):
        raw = self.load_raw_case(index)
        body_id = torch.from_numpy(raw['body_id']).long()
        coords = self.normalizer.apply_x(torch.from_numpy(raw['coords']).float())
        load = self.normalizer.apply_load(torch.from_numpy(raw['prescribed_displacement']).float())
        identity = torch.nn.functional.one_hot(body_id, num_classes=19).float()
        condition = torch.cat([coords, load, identity], dim=-1)
        sol = self.normalizer.apply_y2(torch.from_numpy(raw['sol']).float(), body_id)
        return coords, condition, sol, body_id

    def dim(self):
        return 3, 25, 5

    def get_normalizer(self):
        return self.normalizer

    def get_case_names(self):
        return list(self.case_names)


class RolloutDataset(torch.utils.data.Dataset):

    class Normalizer:
        def __init__(self, x, y2):
            self.x_flag = False
            self.y2_flag = False

            old_x_shape = x.shape
            old_y2_shape = y2.shape

            x = torch.reshape(x, (-1, x.shape[-1]))
            y2 = torch.reshape(y2, (-1, y2.shape[-1]))

            self.x_mean = torch.mean(x, dim=0)
            self.x_std = torch.std(x, dim=0) + 1e-8
            self.y2_mean = torch.mean(y2, dim=0)
            self.y2_std = torch.std(y2, dim=0) + 1e-8

            x = torch.reshape(x, old_x_shape)
            y2 = torch.reshape(y2, old_y2_shape)

        def set(self, x_mean=None, x_std=None, y2_mean=None, y2_std=None, **_):
            if x_mean is not None:
                self.x_mean = x_mean.detach().clone()
            if x_std is not None:
                self.x_std = x_std.detach().clone()
            if y2_mean is not None:
                self.y2_mean = y2_mean.detach().clone()
            if y2_std is not None:
                self.y2_std = y2_std.detach().clone()

        def export(self):
            return {
                "x_mean": self.x_mean.detach().clone(),
                "x_std": self.x_std.detach().clone(),
                "y2_mean": self.y2_mean.detach().clone(),
                "y2_std": self.y2_std.detach().clone(),
            }

        def is_apply_x(self):
            return self.x_flag

        def is_apply_y2(self):
            return self.y2_flag

        def apply_x(self, x, inverse=False):
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)

            old_x_shape = x.shape
            x = torch.reshape(x, (-1, x.shape[-1]))
            if not inverse:
                x = (x - self.x_mean) / self.x_std
                self.x_flag = True
            else:
                x = x * self.x_std + self.x_mean
            return torch.reshape(x, old_x_shape)

        def apply_y2(self, y2, inverse=False):
            self.y2_mean = self.y2_mean.to(y2.device)
            self.y2_std = self.y2_std.to(y2.device)

            old_y2_shape = y2.shape
            y2 = torch.reshape(y2, (-1, y2.shape[-1]))
            if not inverse:
                y2 = (y2 - self.y2_mean) / self.y2_std
                self.y2_flag = True
            else:
                y2 = y2 * self.y2_std + self.y2_mean
            return torch.reshape(y2, old_y2_shape)

    def __init__(self, data_mode, apply_norm=True, data_root="./dataset", norm_params=None):
        super().__init__()
        data_file = os.path.join(data_root, f"{data_mode}.npy")
        dataset = np.load(data_file, allow_pickle=True).tolist()

        self.coords = torch.tensor(dataset["coords"]).float()
        self.sol = torch.tensor(dataset["sol"]).float()

        self.normalizer = RolloutDataset.Normalizer(self.coords, self.sol)
        if norm_params is not None:
            self.normalizer.set(**norm_params)
        if apply_norm:
            self.coords = self.normalizer.apply_x(self.coords)
            self.sol = self.normalizer.apply_y2(self.sol)

        self.length = self.coords.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.sol[idx]

    def dim(self):
        return self.coords.shape[-1], self.sol.shape[-1]

    def get_normalizer(self):
        return self.normalizer

    def total_steps(self):
        return self.sol.shape[-2]


class TimeEmbeddingDataset(torch.utils.data.Dataset):

    def __init__(self, data_mode, apply_norm=True, coords_condition=True, data_root="./dataset", norm_params=None):
        super().__init__()
        data_file = os.path.join(data_root, f"{data_mode}.npy")
        dataset = np.load(data_file, allow_pickle=True).tolist()

        self.coords = torch.tensor(dataset["coords"]).float()
        if coords_condition:
            self.condition = torch.cat((self.coords, torch.tensor(dataset["condition"]).float()), dim=-1)
        else:
            self.condition = torch.tensor(dataset["condition"]).float()
        self.sol = torch.tensor(dataset["sol"]).float()
        self.t = torch.tensor(dataset["t"]).float()

        self.normalizer = StaticDataset.Normalizer(self.coords, self.condition, self.sol)
        if norm_params is not None:
            self.normalizer.set(**norm_params)
        if apply_norm:
            self.coords = self.normalizer.apply_x(self.coords)
            self.condition = self.normalizer.apply_y1(self.condition)
            self.sol = self.normalizer.apply_y2(self.sol)

        self.length = self.coords.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.condition[idx], self.sol[idx], self.t[idx]

    def dim(self):
        return self.coords.shape[-1], self.condition.shape[-1], self.sol.shape[-1]

    def get_normalizer(self):
        return self.normalizer

    def total_steps(self):
        return self.sol.shape[-2]
