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



def save_model(pipeline, evaluation, model_path, metadata_path, best_model_path):
    """
    Save the trained model and its metadata. If it's the best model so far, update the best model.

    Args:
        pipeline (Pipeline): Trained pipeline.
        evaluation (dict): Evaluation results.
        model_path (str): Path to save the trained model.
        metadata_path (str): Path to save metadata.
        best_model_path (str): Path to save the best model.

    Returns:
        None
    """
    # Save current model
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {"evaluation": evaluation}
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)

    # Check if it's the best model
    if not os.path.exists(best_model_path):
        is_best_model = True
    else:
        with open(best_model_path, "r") as file:
            best_metadata = json.load(file)
        is_best_model = evaluation["r2_score"] > best_metadata["evaluation"]["r2_score"]

    if is_best_model:
        joblib.dump(pipeline, best_model_path.replace(".json", ".pkl"))
        with open(best_model_path, "w") as file:
            json.dump(metadata, file)
