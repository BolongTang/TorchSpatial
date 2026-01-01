# SpatialRelationEncoder/ (package)
#   __init__.py  (empty)
#   aodha_ffn.py  (defines class AodhaFFNSpatialRelationLocationEncoder)

import importlib
from typing import Any, Dict, Tuple, Union, Type

# The entries' names are copied from the tutorial.ipynb specification. 
# TODO: Match each encoder entry to the actual file under SpatialRelationEncoder. 
ENCODER_REGISTRY: Dict[str, Union[type, Tuple[str, str]]] = {
    # name: (module_path, class_name)
    "Space2Vec-grid": ("..SpatialRelationEncoder.GridCellSpatialRelationLocationEncoder", "GridCellSpatialRelationLocationEncoder"),
    "Space2Vec-theory": ("..SpatialRelationEncoder.TheoryGridCellSpatialRelationLocationEncoder", "TheoryGridCellSpatialRelationLocationEncoder"),
    "xyz": ("..SpatialRelationEncoder.XYZSpatialRelationLocationEncoder", "XYZSpatialRelationLocationEncoder"),
    "NeRF": ("..SpatialRelationEncoder.NERFSpatialRelationLocationEncoder", "NERFSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereC": ("..SpatialRelationEncoder.", ""),
    "Sphere2Vec-sphereC+": ("..SpatialRelationEncoder.", ""),
    "Sphere2Vec-sphereM": ("..SpatialRelationEncoder.", ""),
    "Sphere2Vec-sphereM+": ("..SpatialRelationEncoder.", ""),
    "Sphere2Vec-dfs": ("..SpatialRelationEncoder.", ""),
    "rbf": ("..SpatialRelationEncoder.RBFSpatialRelationLocationEncoder", "RBFSpatialRelationLocationEncoder"),
    "rff": ("..SpatialRelationEncoder.RFFSpatialRelationLocationEncoder", "RFFSpatialRelationLocationEncoder"),
    "wrap": ("..SpatialRelationEncoder.", ""),
    "wrap_ffn": ("..SpatialRelationEncoder.", ""),
    "tile_ffn": ("..SpatialRelationEncoder.", "")
}

def _resolve_encoder(name: str) -> type:
    entry = ENCODER_REGISTRY[name]
    if isinstance(entry, type):
        return entry  # already resolved / cached

    module_path, class_name = entry
    mod = importlib.import_module(module_path, package="TorchSpatial.modules")
    cls = getattr(mod, class_name) # cls means the actual class object

    ENCODER_REGISTRY[name] = cls  # cache for next time
    return cls

def get_loc_encoder(name: str, overrides: Dict[str, Any] | None = None, **kwargs):
    EncoderCls = _resolve_encoder(name)
    cfg = {**kwargs, **(overrides or {})} # **kwargs implies no need to hard code encoder-specific parameters
    return EncoderCls(**cfg)

