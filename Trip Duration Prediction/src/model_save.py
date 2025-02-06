import json
import os
import joblib

def save_evaluation_results(evaluation, output_path, model_id):
    """
    Save evaluation results to a JSON file, including the model ID.

    Args:
        evaluation (dict): Evaluation results.
        output_path (str): Path to save evaluation results.
        model_id (str): ID of the model.

    Returns:
        None
    """
    evaluation["model_id"] = model_id
    with open(output_path, "w") as file:
        json.dump(evaluation, file)


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
