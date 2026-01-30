import os

try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyYAML is required. Install with: pip install pyyaml")


def load_config(dataset_name: str):
    """Load dataset config YAML. Tries package-relative path then common fallbacks."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "configs", f"{dataset_name}.yaml"),
        os.path.join(os.getcwd(), "ML", "configs", f"{dataset_name}.yaml"),
        os.path.join(os.getcwd(), "configs", f"{dataset_name}.yaml"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(f"Config file for dataset '{dataset_name}' not found. Tried: {candidates}")


def available_datasets():
    base = os.path.join(os.path.dirname(__file__), "configs")
    if not os.path.isdir(base):
        return []
    return [p[:-5] for p in os.listdir(base) if p.endswith(".yaml")]
