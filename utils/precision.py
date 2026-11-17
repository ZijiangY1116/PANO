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

"""Runtime AMP policy shared by training and evaluation (weights stay FP32)."""
import torch



def model_forward(model, coords, condition, t=None, mixed_precision="none"):
    if mixed_precision not in ("none", "fp16"):
        raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")
    enabled = mixed_precision == "fp16"
    if enabled and coords.device.type != "cuda":
        raise ValueError("FP16 mixed precision requires CUDA.")
    with torch.autocast(device_type=coords.device.type, dtype=torch.float16, enabled=enabled):
        result = model(coords, condition) if t is None else model(coords, condition, t)
    # Physical denormalization, losses and metrics stay in FP32.
    return result.float() if enabled else result


def make_scaler(mixed_precision):
    if mixed_precision not in ("none", "fp16"):
        raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")
    return torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")


def backward_step(loss, model, optimizer, scaler, grad_clip):
    scaler.scale(loss).backward()
    return step_accumulated_gradients(model, optimizer, scaler, grad_clip)


def step_accumulated_gradients(model, optimizer, scaler, grad_clip):
    """Unscale, clip and update once after all scaled backward calls."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # A reduced scale indicates that overflow caused the optimizer step to skip.
    return scaler.get_scale() >= scale_before
