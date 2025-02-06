from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def initialize_pipeline(numeric_features, categorical_features, pipeline_steps, hyperparameters):
    """
    Initialize a Ridge model pipeline with configurable preprocessing steps.

    Args:
        features (list): List of feature names.
        pipeline_steps (list): List of preprocessing steps (e.g., scaling, encoding).
        hyperparameters (dict): Ridge model hyperparameters.

    Returns:
        Pipeline: Configured machine learning pipeline.
    """
    features = numeric_features + categorical_features
    
    transformers = []

    #setting transformers
    if "StanderScaler" in pipeline_steps:
        transformers.append(("scaler", StandardScaler(), numeric_features))
    if "MinMaxScaler" in pipeline_steps:
        transformers.append(("scaler", MinMaxScaler(), numeric_features))
    if "encoding" in pipeline_steps:
        transformers.append(("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features))
        
        
    preprocessor = ColumnTransformer(transformers, remainder="passthrough")
    
    model = Ridge(**hyperparameters)
    
    #pipeline creation
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline

def train_and_evaluate(pipeline,config, train_df, val_df):
    X_train = train_df.drop(config["target"], axis=1)
    y_train = train_df[config["target"]]
    X_val = val_df.drop(config["target"], axis=1)
    y_val = val_df[config["target"]]


    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    evaluation_results = {
        "mean_squared_error": mean_squared_error(y_val, y_pred),
        "r2_score": r2_score(y_val, y_pred)
    }
    
    return pipeline, evaluation_results
