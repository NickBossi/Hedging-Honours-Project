import numpy as np
import pandas as pd
from utils.leadlag import leadlag, undo_leadlag
import torch
import torch.nn.functional as F
import signatory
from InvertSignatorySignatures import invert_signature
import matplotlib.pyplot as plt
import pickle
import sys
import json
from datetime import datetime
import re
import wandb
import os


def sanitize_filename(filename):
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '-', filename)
    # Optionally, truncate the file name if it's too long
    return filename[:200]


def test_size():

    depth = 10

    # File paths for training and testing sets
    train_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\train_stock_data.csv"
    test_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\test_stock_data.csv"

    # Gets training and testing data
    traindf = pd.read_csv(train_path)
    testdf = pd.read_csv(test_path)

    # Performs transpose and adds batch dimension for signature
    train_data = torch.unsqueeze(torch.tensor(traindf.values), 0)
    test_data = torch.unsqueeze(torch.tensor(testdf.values), 0)

    #print(train_data.shape)

    # Initiating empty list which will hold all signatures
    train_signature_data = []
    test_signature_data = []
    rolling_train_data = []

    # Getting dimensions of data to be looped over
    n = train_data.shape[1]
    #print(n)
    m = test_data.shape[1]

    num_samples = int(n/depth)
    tempdf = train_data[:,0:depth,:]
    print(tempdf)
    tempdf = leadlag(tempdf)
    print(tempdf)
    print(tempdf.shape)

    #signature = signatory.signature(path = tempdf, depth = depth)
    #print(signature.shape)
    #inverse = invert_signature(signature=signature, depth=depth, channels = 4)[:,1:,]

def test_inversion(data_config):
    depth = data_config['depth']
    num_channels = data_config['num_stocks']
    test_paths = []
    #test_signatures = np.load('../final_VAE/data/generated_test_sigs.npy')
    test_signatures = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_test_sigs.npy')
    test_signatures = torch.tensor(test_signatures, dtype=torch.float32)
    for signature in test_signatures:
        signature = torch.unsqueeze(signature, 0)                                                                       #add batch dimension for signature inversion
        path = invert_signature(signature, depth = depth, channels = num_channels).squeeze(0) 
        test_paths.append(path)
    #np.save('data/np_generated_time_series', test_paths)
    print("Inverse test paths GENERATED!")
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/np_generated_time_series', test_paths)

def signature_inversion(data_config):
    generated_paths = []

    #signature_paths = np.load('../final_VAE/data/generated_sigs.npy')
    signature_paths = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_sigs.npy')

    signature_paths = torch.tensor(signature_paths, dtype=torch.float32)


    window_size = data_config['window_size']
    depth = data_config['depth']
    num_stocks = data_config['num_stocks']
    num_channels = num_stocks

    new_depth = data_config['window_size']*2-1

    if data_config['leadlag']:
        num_channels = 2*num_stocks

    signature_size = signatory.signature_channels(channels = num_channels, depth = depth)
    new_signature_size = signatory.signature_channels(channels = num_channels, depth = new_depth)

    diff = new_signature_size - signature_size

   


    for signature_path in signature_paths:
        #dealing with initial path

        first_path = torch.unsqueeze(signature_path[0], 0)

        path = invert_signature(first_path, depth = depth, channels = num_channels).squeeze(0) 

        path = (path - path[0,:]).numpy()
        if data_config["leadlag"]:
            path = undo_leadlag(path)

        for signature in signature_path[1:]:

            signature = torch.unsqueeze(signature, 0)                                                                       #add batch dimension for signature inversion
           
            inverted_signature = invert_signature(signature, depth = depth, channels = num_channels).squeeze(0)                        

            inverted_signature = (inverted_signature - inverted_signature[0,:]).numpy()

            if config["leadlag"]:
                inverted_signature = undo_leadlag(inverted_signature)                       #lowers to zero and undoes lead_lag transformation

            inverted_signature = (inverted_signature + path[-1,:])[1:]                                                #shifts path to start at end of previous path
            path = np.concatenate((path, inverted_signature), axis = 0)

        generated_paths.append(path)

    generated_paths = np.array(generated_paths)[:,:data_config["forecast_horizon"]+1,:]
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_paths.npy', generated_paths)

    return generated_paths

def plot_paths(config_str, config):


    if config["wandb"]:
        run_id_file = "d:/Documents/UCT/Honours/Honours_Project/Code/data/run_id.txt"
        #run_id_file = os.path.join(os.path.dirname(__file__), "data", "run_id.txt")

        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
            
        wandb.init(project = "Final_Signature_Generation", id = run_id, resume = "must")

    generated_paths = signature_inversion(config)
    num_paths = len(generated_paths)
    num_dimensions =  len(generated_paths[0])
    colors = plt.cm.jet(np.linspace(0,1,num_paths))
    fig = plt.figure(figsize=(30, 20))
    for i, path in enumerate(generated_paths):
        if config["num_paths"] == 1:
            plt.plot(path, color = colors[i])
        else:
            plt.plot(path[:,0], color = "blue")
            plt.plot(path[:,1], color = "orange")
        

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
    title = sanitize_filename(config_str)+'_'+timestamp
    plt.legend(loc='upper left', title = title, fontsize='medium', shadow=True, frameon=True)
    print("Got here")
    plt.savefig('plots/{}.png'.format(title))

    if config['show_plot']:
        plt.show()
    if config["wandb"]:
        wandb.log({"plot": fig})
        wandb.finish()
'''
signature = torch.unsqueeze(signature, 0)                                                                       #add batch dimension for signature inversion
#print(signature.shape)
inverted_signature = invert_signature(signature, depth = 7, channels = 4).squeeze(0)[1:]                        #removes first element
inverted_signature = undo_leadlag((inverted_signature - inverted_signature[0,:]).numpy())                       #lowers to zero and undoes lead_lag transformation
generated_paths.append(inverted_signature)
#print(inverted_signature) 
#print(inverted_signature.shape)
plt.plot(inverted_signature)
plt.show()
'''

if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)

    config_str = config_str.replace(' ', '_').replace('{', '').replace('}', '').replace(':', '-').replace(',', '_').replace('"', '')
    test_inversion(config)
    plot_paths(config_str, config)
    print("Path inversion and plotting complete.")

