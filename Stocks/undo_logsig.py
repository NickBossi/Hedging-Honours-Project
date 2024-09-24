import iisignature
import numpy as np
import pandas as pd
from utils.leadlag import leadlag
import torch
import sys
import json
import pickle

# File paths for training and testing sets
train_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/train_stock_data.csv"
test_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/test_stock_data.csv"

# Gets training and testing data
train_data = pd.read_csv(train_path)    #shape = (9610,2)
test_data = pd.read_csv(test_path)

n = train_data.shape[0]
dimension = train_data.shape[1]
print("Dimension of training data after leadlag is performed:{}".format(dimension))
m = test_data.shape[0]

train_signature_data = []
test_signature_data = []


def get_s(config):
    window_size = config['window_size']
    depth = config['depth']
    #depth = 2*window_size-1
    if config["leadlag"]:
        s = iisignature.prepare(dimension*2, depth, "S2")
    else:
        s = iisignature.prepare(dimension, depth, "S2")

    return s

def undo_log_sig(config):

    #Load prepared signature
    s = get_s(config)

    logsig_paths = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_logsigs.npy', allow_pickle=True)
    logsig_test_paths = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_test_logsigs.npy', allow_pickle=True)
    # logsig_paths = np.load('../final_VAE/data/generated_logsigs.npy', allow_pickle=True)
    # logsig_test_paths = np.load('../final_VAE/data/generated_test_logsigs.npy', allow_pickle=True)
    #print(logsig_paths.shape)

    signatures = []
    test_signatures = []

    
    # Iterates through generated log signatures and converts them back to signatures
    for logsig_path in logsig_paths:
        sig_path = []
        for logsig in logsig_path:
            sig = iisignature.logsigtosig(logsig, s)
            sig_path.append(sig)
        signatures.append(np.array(sig_path))
    signatures = np.array(signatures)
    #print(signatures.shape)
    #Saves generated signatures
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_sigs.npy', signatures)

    #Iterates through the log signatures which were generated over same period as original data and converts to signatures
    for logsig_path in logsig_test_paths:
        sig_path = []
        for logsig in logsig_path:
            sig = iisignature.logsigtosig(logsig, s)
            sig_path.append(sig)
        test_signatures.append(np.array(sig_path))

    test_signatures = np.squeeze(test_signatures, axis=1)
    
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_test_sigs.npy', test_signatures)
        

if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)
    undo_log_sig(config)
    print("Undoing log signatures complete.")
    
