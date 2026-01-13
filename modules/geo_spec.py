import torch
from dataclasses import dataclass

@dataclass
class GeoSpec:
    coord_order: tuple = ("lat", "lon")
    bounds: dict = None          # {"lat": (-90, 90), "lon": (-180, 180)}
    normalize: bool = True       # map to [-1, 1]

    def transform(self, raw_coords: torch.Tensor) -> torch.Tensor:
        """
        raw_coords: (B, 2) in dataset-native order
        returns: canonical coords (B, 2)
        """
        coords = raw_coords.clone()

        if self.normalize and self.bounds is not None:
            for i, key in enumerate(self.coord_order):
                lo, hi = self.bounds[key]
                coords[:, i] = 2 * (coords[:, i] - lo) / (hi - lo) - 1

        return coords

    def to_encoder_input(self, canon_coords: torch.Tensor, encoder_name: str):
        """
        Optional adapter for encoders with special expectations.
        """
        if encoder_name == "space2vec_grid":
            return canon_coords  # already what it expects
        raise ValueError(f"No adapter for encoder {encoder_name}")

# A `GeoSpec` should be a **small, declarative transformer**: it defines what coordinates mean and provides deterministic transforms between raw, canonical, and encoder-specific formats—nothing model-related.

# ### Example `GeoSpec`

# ```python
# import torch
# from dataclasses import dataclass

# @dataclass
# class GeoSpec:
#     coord_order: tuple = ("lat", "lon")
#     bounds: dict = None          # {"lat": (-90, 90), "lon": (-180, 180)}
#     normalize: bool = True       # map to [-1, 1]

#     def transform(self, raw_coords: torch.Tensor) -> torch.Tensor:
#         """
#         raw_coords: (B, 2) in dataset-native order
#         returns: canonical coords (B, 2)
#         """
#         coords = raw_coords.clone()

#         if self.normalize and self.bounds is not None:
#             for i, key in enumerate(self.coord_order):
#                 lo, hi = self.bounds[key]
#                 coords[:, i] = 2 * (coords[:, i] - lo) / (hi - lo) - 1

#         return coords

#     def to_encoder_input(self, canon_coords: torch.Tensor, encoder_name: str):
#         """
#         Optional adapter for encoders with special expectations.
#         """
#         if encoder_name == "space2vec_grid":
#             return canon_coords  # already what it expects
#         raise ValueError(f"No adapter for encoder {encoder_name}")
# ```

# ### What this achieves

# * **Single source of truth** for coordinate meaning (units, order, bounds).
# * **Canonical interface**: all models see the same `(B, coord_dim)` input.
# * **Extensible**: new encoders add adapters only if they truly need a new convention.
