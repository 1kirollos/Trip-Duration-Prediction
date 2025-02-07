import json
import os
import joblib

import json


import json
import os

def save_metadata(evaluation, metadata, output_path):
    """
    Append evaluation results and model metadata to a JSON file.

    Args:
        evaluation (dict): Evaluation results.
        metadata (dict): Model metadata (should include model_id).
        output_path (str): Path to save evaluation results.

    Returns:
        None
    """
    new_entry = {
        "evaluation": evaluation,
        "metadata": metadata  # Assumes metadata already contains 'model_id'
    }
    
    # Load existing data if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):  
                    existing_data = [existing_data]  # Convert to list if it's a single dictionary
            except json.JSONDecodeError:
                existing_data = []  # Handle cases where the file is empty or corrupted
    else:
        existing_data = []

    # Append new data
    existing_data.append(new_entry)

    # Save updated data
    with open(output_path, "w") as file:
        json.dump(existing_data, file, indent=4)



###############################################################

def overwrite_metadata(evaluation, metadata, output_path):
    """
    Save evaluation results and model metadata to a JSON file.

    Args:
        evaluation (dict): Evaluation results.
        metadata (dict): Model metadata (should include model_id).
        output_path (str): Path to save evaluation results.

    Returns:
        None
    """
    results = {
        "evaluation": evaluation,
        "metadata": metadata  # Assumes metadata already contains 'model_id'
    }
    
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)



def save_model(pipeline, directory_path, model_id="model"):
    """
    Saves the trained model pipeline as a .pkl file in the specified directory.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): The trained model pipeline.
        directory_path (str): The directory where the model should be saved.
        model_id (str): The name of the saved model file (without extension).
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Define the full file path
    file_path = os.path.join(directory_path, f"{model_id}.pkl")

    # Save the pipeline to the file
    joblib.dump(pipeline, file_path)

    print(f"Model saved to {file_path}")



def delete_all_files(directory_path):
    """
    Deletes all files in the specified directory.

    Parameters:
        directory_path (str): The directory whose files should be deleted.
    """
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            # Only delete files (not subdirectories)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    else:
        print(f"Directory {directory_path} does not exist.")

import json

def load_metadata(metadata_path):
    """
    Load and simplify model configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Simplified configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)

    metadata = config.get("metadata", {})

    return {
        "model_id": metadata.get("model_id", "model"),
        "numeric_features": metadata.get("numeric_features", []),
        "categorical_features": metadata.get("categorical_features", []),
        "target": metadata.get("target", "target"),
        "pipeline_steps": metadata.get("pipeline_steps", []),
        "hyperparameters": metadata.get("hyperparameters", {}),
        "evaluation": config.get("evaluation", {})
    }


def save_best_model(pipeline, evaluation, metadata, metadata_path, best_model_path):

    metadata = load_metadata(metadata_path)
    
    r2_score = metadata.get("evaluation", {}).get("r2_score", None)

    #if is_better(evaluation, metadata_path):

    overwrite_metadata(evaluation, metadata, metadata_path)
    delete_all_files(best_model_path)
    save_model(pipeline, best_model_path, metadata["model_id"])