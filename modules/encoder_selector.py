import importlib
from typing import Any, Dict, Tuple, Union, Type

# The entries' names are copied from the documentation.
ENCODER_REGISTRY: Dict[str, Union[type, Tuple[str, str]]] = {
    # name: (module_path, class_name)
    "Space2Vec-grid": ("..SpatialRelationEncoder.GridCellSpatialRelationLocationEncoder", "GridCellSpatialRelationLocationEncoder"),
    "Space2Vec-theory": ("..SpatialRelationEncoder.TheoryGridCellSpatialRelationLocationEncoder", "TheoryGridCellSpatialRelationLocationEncoder"),
    "xyz": ("..SpatialRelationEncoder.XYZSpatialRelationLocationEncoder", "XYZSpatialRelationLocationEncoder"),
    "NeRF": ("..SpatialRelationEncoder.NERFSpatialRelationLocationEncoder", "NERFSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereC": ("..SpatialRelationEncoder.SphereSpatialRelationLocationEncoder", "SphereSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereC+": ("..SpatialRelationEncoder.SphereGridSpatialRelationLocationEncoder", "SphereGridSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereM": ("..SpatialRelationEncoder.SphereMixScaleSpatialRelationLocationEncoder", "SphereMixScaleSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereM+": ("..SpatialRelationEncoder.SphereGridMixScaleSpatialRelationLocationEncoder", "SphereGridMixScaleSpatialRelationLocationEncoder"),
    "Sphere2Vec-dfs": ("..SpatialRelationEncoder.DFTSpatialRelationLocationEncoder", "DFTSpatialRelationLocationEncoder"),
    "rbf": ("..SpatialRelationEncoder.RBFSpatialRelationLocationEncoder", "RBFSpatialRelationLocationEncoder"),
    "rff": ("..SpatialRelationEncoder.RFFSpatialRelationLocationEncoder", "RFFSpatialRelationLocationEncoder"),
    "wrap": ("..modules.models", "FCNet"),
    "wrap_ffn": ("..SpatialRelationEncoder.AodhaFFNSpatialRelationLocationEncoder", "AodhaFFNSpatialRelationLocationEncoder"),
    "tile_ffn": ("..SpatialRelationEncoder.GridLookupSpatialRelationLocationEncoder", "GridLookupSpatialRelationLocationEncoder"),
    "Siren(SH)": ("..SpatialRelationEncoder.SphericalHarmonicsSpatialRelationLocationEncoder", "SphericalHarmonicsSpatialRelationLocationEncoder"),
    # These below are not in the allowed options in tutorial.ipynb but Nemin says they can be implemented
    # Not supported by the current training implementation
    "GridCellNorm": ("..SpatialRelationEncoder.GridCellNormSpatialRelationEncoder", "GridCellNormSpatialRelationEncoder"),
    "HexagonGridCell": ("..SpatialRelationEncoder.HexagonGridCellSpatialRelationEncoder", "HexagonGridCellSpatialRelationEncoder"),
    "Naive": ("..SpatialRelationEncoder.NaiveSpatialRelationEncoder", "NaiveSpatialRelationEncoder"),
    "TheoryDiagGridCellSpatialRelationEncoder": ("..SpatialRelationEncoder.TheoryDiagGridCellSpatialRelationEncoder", "TheoryDiagGridCellSpatialRelationEncoder"),
    "": ("..SpatialRelationEncoder.", ""),
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

