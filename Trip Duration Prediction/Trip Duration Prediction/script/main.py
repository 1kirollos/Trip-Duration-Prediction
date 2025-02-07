import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_utils import *
from model_config import *
from model_train import *   
from model_save import *
import numpy as np
import pandas as pd

train_df = prepare_data("Data/raw/test/train.csv", remove_outliers_flag=False)
val_df = prepare_data("Data/raw/test/val.csv", remove_outliers_flag=False)

# save_data(train_df, "Data/processed/test/train.csv")
# save_data(val_df, "Data/processed/test/val.csv")


config = load_config("config\model_config.json")
print(config)

pipeline = initialize_pipeline(config["numeric_features"], config["categorical_features"], config["pipeline_steps"], config["hyperparameters"])




model, evaluation = train_and_evaluate(pipeline, config, train_df, val_df)
print(evaluation)

save_metadata(evaluation, config,"models\model_versioning\models_history.json")
#save_model(model, evaluation, )