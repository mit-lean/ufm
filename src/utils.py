import os
import torch
import yaml 

def initiate_parameters(path_to_yaml):
    # read in yaml as directory
    with open(path_to_yaml) as f: 
        run_parameters = yaml.load(f, Loader=yaml.FullLoader)
    return run_parameters 

def get_trained_model_directory(run_parameters, ensemble_member_number):
    output_directory = os.path.join(run_parameters['trained_models_path'],run_parameters['trained_model_folder'])+str(ensemble_member_number)
    return output_directory