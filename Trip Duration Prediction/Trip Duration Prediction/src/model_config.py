import json

def load_config(config_path):
    """
    Load and simplify model configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Simplified configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)

    return {
        "model_id" : config.get("model_id", "model"),
        "numeric_features": config["model"].get("numeric_features", []),
        "categorical_features": config["model"].get("categorical_features", []),
        "target": config["model"].get("target", "target"),
        "pipeline_steps": config["model"].get("pipeline_steps", []),
        "hyperparameters": config["model"].get("hyperparameters", {})
    }
