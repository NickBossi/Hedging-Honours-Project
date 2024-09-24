import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import json


def plot_data(data):
    plt.plot(data)
    plt.show()

def load_data():
    #Loading csv's

    apple_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\AAPL.csv"
    microsoft_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\MSFT.csv"

    appledf = pd.read_csv(apple_path)
    microdf = pd.read_csv(microsoft_path)

    #Truncating apple data rows to ensure same length as microsoft data
    appledf = appledf.tail(microdf.shape[0])

    #Extracting only the opening price from the stocks
    applecolumn = appledf['Open'].values
    microcolumn = microdf['Open'].values

    #Creating a dataframe of the stocks
    file = pd.DataFrame(np.stack((applecolumn, microcolumn), axis=1), columns=['apple', 'microsoft'])

    #plt.plot(applecolumn[-40:])
    #plt.show()
    #plt.plot(microcolumn[-40:])
    #plt.show()
    return file


def save_data(file):

    # file_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\stock_data.csv"
    # train_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\train_stock_data.csv"
    # test_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\test_stock_data.csv"
    file_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/stock_data.csv"
    train_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/train_stock_data.csv"
    test_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/test_stock_data.csv"

    #Splitting the data into training and testing sets
    test_stock_data = file.tail(30)
    train_stock_data = file.iloc[:-30]

    #Saving the data to csv files
    file.to_csv(file_path, index=False)
    train_stock_data.to_csv(train_path, index=False)
    test_stock_data.to_csv(test_path, index=False)




def main():
    #Gets config from master programme
    config_str = sys.argv[1]
    data_load_config = json.loads(config_str)
    num_stocks = data_load_config['num_stocks']
    num_days = data_load_config['num_days']

    #Loads data
    data = load_data()

    #Truncates data to the specified number of stocks and days
    if num_stocks < data.shape[1]:
        data = data.iloc[:,0:num_stocks]
    else:
        print("All stocks being used")

    if num_days <= data.shape[0]:
        data = data.tail(num_days)
    
    #Saves data
    save_data(data)
    
    print("Done loading and saving data.")

if __name__ == "__main__":
    main()
    




    
