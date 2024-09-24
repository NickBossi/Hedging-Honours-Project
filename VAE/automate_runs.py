import subprocess
import json
import itertools




configs = {
    "alpha": [0.001],
    "latent_dim": [8],
    "n_hidden": [100],
    "num_days": [1000],

    "lr": [3e-4],

    "batch_size": [16],

    "n_epochs": [10000],

    "num_stocks": [2],

    "window_size": [5],
    "depth": [5],
    "forecast_horizon": [31],
    "num_paths": [100000],
    "variance": [1],
    "dropout": [0],

    "plot_latent": [False],

    "show_plot": [False],

    "train_model": [False],

    "generate_paths": [True],

    "generate_latent": [False],

    "leadlag": [False],

    "wandb": [False]
    
}


'''

hyperparameters ={
    "latent_dim": [4, 8,16],
    "lr": [0.005, 0.0005],
    "n_epochs": [1000],
    "batch_size": [64, 512, 1024],
    "alpha": [0.3,0.03,0.003],
    "dropout": [0],
}
'''

config_list = list(iter(itertools.product(*configs.values())))

data_creation_path = r'd:\Documents\UCT\Honours\Honours_Project\Code\Stocks\dataprocessing.py'
to_logsig_path = r'd:\Documents\UCT\Honours\Honours_Project\Code\Stocks\to_logsig.py'
cvae_path = r'd:\Documents\UCT\Honours\Honours_Project\Code\final_VAE\cvae.py'
undo_logsig_path = r'd:\Documents\UCT\Honours\Honours_Project\Code\Stocks\undo_logsig.py'
inversion_path = r'd:\Documents\UCT\Honours\Honours_Project\Code\Stocks\signature_inversion_and_plotting.py'
discriminator_path = r'd:\Documents\UCT\Honours\Honours_Project\Code\final_VAE\process_discriminator.py'


python_env1 = r'c:\Users\Nick\AppData\Local\Programs\Python\Python312\python.exe'
python_env2 = r'c:\Users\Nick\anaconda3\envs\iisignature_test\python.exe'
python_env3 = r'c:\Users\Nick\miniconda3\python.exe'

def run_script(python_executable, script, args):
    subprocess.run([python_executable, '-Xfrozen_modules=off', script] + args)

   
for item in config_list:

    #Converting configs to strings
    config = dict(zip(configs.keys(), item))
    config_str = json.dumps(config)
    
    if config["train_model"]:
        run_script(python_env1, data_creation_path, [config_str])
        run_script(python_env2, to_logsig_path, [config_str])
    if config["generate_paths"] or config["train_model"] or config["generate_latent"]:
        run_script(python_env1, cvae_path, [config_str])
    if config["generate_paths"]:
        run_script(python_env2, undo_logsig_path, [config_str])
        run_script(python_env3, inversion_path, [config_str])
