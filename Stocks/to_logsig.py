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
#print(dimension)
m = test_data.shape[0]

discriminator_sigs = []
train_signature_data = []
test_signature_data = []


def get_s(config):

    depth = config['depth']
    #depth = 2*window_size-1
    if config["leadlag"]:
        s = iisignature.prepare(dimension*2, depth, "S2")
    else:
        s = iisignature.prepare(dimension, depth, "S2")

    return s

def create_signature(config):
    window_size = config['window_size']
    depth = config['depth']

    for i in range(int(n/window_size)):
        temp_data = train_data[i*window_size:(i+1)*window_size]        # gets depth number of days and applies leadlag transformation
        if config["leadlag"]:
            temp_data = leadlag(temp_data)

        #print(temp_data.shape)
        sig = iisignature.sig(temp_data, depth)                    # gets signature
        #print(sig.shape)
    
        discriminator_sigs.append(sig)          
                                                
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/discriminator_sigs.npy', discriminator_sigs)

def create_log_sig(config):
    s = get_s(config)
    window_size = config['window_size']
    depth = config['depth']
    #depth = 2*window_size-1

    for i in range(int(n/window_size)):
        temp_data = train_data[i*window_size:(i+1)*window_size]        # gets window_size number of days and applies leadlag transformation
        if config["leadlag"]:
            temp_data = leadlag(temp_data)
        #print(temp_data.shape)
        #sig = iisignature.sig(temp_data, depth)                    # gets signature
        #print(sig.shape)
        logsignature = iisignature.logsig(temp_data, s)                # gets log signature
        #print(logsignature.shape)
        train_signature_data.append(logsignature)                                                  
    
    for j in range(int(m/window_size)):
        temp_data = test_data[j*window_size:(j+1)*window_size]        # gets depth number of days and applies leadlag transformation
        if config["leadlag"]:
            temp_data = leadlag(temp_data)
        logsignature = iisignature.logsig(temp_data, s)                # gets log signature
        #print(signature.shape)
        test_signature_data.append(logsignature)
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/training_data.npy', train_signature_data)
    np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/testing_data.npy', test_signature_data)



if __name__ == "__main__":
    config_str = sys.argv[1]
    config = json.loads(config_str)
    #create_signature(config)
    create_log_sig(config)
    print("Creation of log signatures complete.")
    
