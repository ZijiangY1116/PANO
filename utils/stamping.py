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

"""Material-wise Stamping metrics; geometry during training is displacement."""
import torch


def stamping_relative_errors(pred, target, body_id, geometry_name='geometry'):
    errors = {}
    for material, name in enumerate(('metal', 'rubber')):
        for quantity, channels in ((geometry_name, slice(0, 3)), ('stress', slice(3, 4)), ('peeq', slice(4, 5))):
            sample_errors = []
            for p, y, ids in zip(pred, target, body_id):
                mask = ids == material
                if not mask.any():
                    raise ValueError(f'Stamping sample has no {name} points.')
                p, y = p[mask, channels].float(), y[mask, channels].float()
                sample_errors.append(torch.linalg.vector_norm(p-y) / torch.linalg.vector_norm(y).clamp_min(1e-8))
            errors[f'{name}_{quantity}_rel_l2'] = torch.stack(sample_errors).mean()
    return errors
