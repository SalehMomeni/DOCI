_MODEL_REGISTRY = {
    "r3m": ("models.R3M", "MetaWorldR3M"),
    "ibc": ("models.IBC", "MetaWorldIBC"),
    "bet": ("models.BeT", "MetaWorldBeT"),
    "dp": ("models.DP", "MetaWorldDiffusionPolicy"),
    "act": ("models.ACT", "MetaWorldACT"),
    "doci": ("models.DOCI", "ESNRidgeBC"),
}


def load_model(name: str, **kwargs):
    """
    Load a MetaWorld model policy.
    """
    if not isinstance(name, str):
        raise TypeError("Model name must be a string")

    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available models: "
            f"{sorted(set(_MODEL_REGISTRY.keys()))}"
        )

    module_name, class_name = _MODEL_REGISTRY[key]

    try:
        module = __import__(module_name, fromlist=[class_name])
        model_cls = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import model '{name}' from '{module_name}'. Original error: {e}"
        ) from e
    except AttributeError as e:
        raise ImportError(
            f"Module '{module_name}' does not define '{class_name}'."
        ) from e

    return model_cls(**kwargs)