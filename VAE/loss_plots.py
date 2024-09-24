import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path1 = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_total_loss.csv'
file_path2 = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_MSE_loss.csv'
file_path3 = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_KL_loss.csv'
ffNN_train_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_ffNN_training_loss.csv'
ffNN_test_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_ffNN_test_loss.csv'

untuned_900_test_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/untuned_LSTM_test_900_loss.csv'
untuned_100_test_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/untuned_LSTM_test_100_loss.csv'
untuned_900_train_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/untuned_LSTM_train_900_loss.csv'
untuned_100_train_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/untuned_LSTM_train_100_loss.csv'


LSTM_train_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_LSTM_training_loss.csv'
LSTM_test_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/wandb_LSTM_test_loss.csv'


def LSTM_loss_plots():
    LSTM_train = pd.read_csv(LSTM_train_path)
    LSTM_test = pd.read_csv(LSTM_test_path)
    x_column = 'Step'
    y_column1 = LSTM_train.columns[1]
    y_column2 = LSTM_test.columns[1]
    plt.figure(figsize=(10,6))

    plt.plot(LSTM_train[x_column], LSTM_train[y_column1], color = 'blue', label = 'Training Loss')
    plt.plot(LSTM_test[x_column], LSTM_test[y_column2], color = 'red', label = 'Test Loss')
    #plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Loss vs Epoch')
    plt.legend()
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/LSTM_loss_plot.png', dpi = 300)
    plt.clf()


def KL_loss_plots():
    total = pd.read_csv(file_path1)
    mse = pd.read_csv(file_path2)
    kl = pd.read_csv(file_path3)
    # Assuming you want to plot 'Step' vs 'Loss', change the column names as necessary
    x_column = 'Step'  # Replace with actual column name for steps/iterations
    y_column = total.columns[1]
    #y_column = 'lyric-shadow-89 - Epoch Loss'  # Replace with actual column name for loss/metric

    # Plot the data on a log scale
    plt.figure(figsize=(10,6))
    plt.plot(total[x_column], total[y_column])

    # Set log scale for y-axis
    plt.yscale('log')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (Log Scale)')
    plt.title(f'{'Total Loss'} vs {'Epoch'} (Log Scale)')
    plt.legend()
    plt.savefig('final_plots/VAE_total_loss_plot.png', dpi = 300)
    plt.clf()


    ######################################################################################################################################################


    # Assuming you want to plot 'Step' vs 'Loss', change the column names as necessary
    x_column = 'Step'  # Replace with actual column name for steps/iterations
    #y_column = 'lyric-shadow-89 - MSE_Loss'  # Replace with actual column name for loss/metric
    y_column = mse.columns[1]

    # Plot the data on a log scale
    plt.figure(figsize=(10,6))
    plt.plot(mse[x_column], mse[y_column])

    # Set log scale for y-axis
    plt.yscale('log')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title(f'{'MSE Loss'} vs {'Epoch'} (Log Scale)')
    plt.legend()
    plt.savefig('final_plots/VAE_MSE_loss_plot.png', dpi = 300)
    plt.clf()


    ######################################################################################################################################################


    # Assuming you want to plot 'Step' vs 'Loss', change the column names as necessary
    x_column = 'Step'  # Replace with actual column name for steps/iterations
    #y_column = 'lyric-shadow-89 - KL_Loss'  # Replace with actual column name for loss/metric
    y_column = kl.columns[1]

    # Plot the data on a log scale
    plt.figure(figsize=(10,6))
    plt.plot(kl[x_column], kl[y_column])

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss')
    plt.title(f'{'KL Loss'} vs {'Epoch'} (Log Scale)')
    plt.legend()
    plt.savefig('final_plots/VAE_KL_loss_plot.png', dpi = 300)

def ffNN_loss_plots():
    ffNN_train = pd.read_csv(ffNN_train_path)
    ffNN_test = pd.read_csv(ffNN_test_path)
    x_column = 'Step'
    y_column1 = ffNN_train.columns[1]
    y_column2 = ffNN_test.columns[1]
    plt.figure(figsize=(10,6))
    #plt.yscale('log')
    plt.plot(ffNN_train[x_column], ffNN_train[y_column1], color = 'blue', label = 'Training Loss')
    plt.plot(ffNN_test[x_column], ffNN_test[y_column2], color = 'red', label = 'Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Feed-Forward Loss vs Epoch')
    plt.legend()
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/ffNN_loss_plot.png', dpi = 300)
    plt.show()
    print()
    
def untuned_loss_plots():
    untuned_900_test = pd.read_csv(untuned_900_test_path)
    untuned_100_test = pd.read_csv(untuned_100_test_path)
    untuned_900_train = pd.read_csv(untuned_900_train_path)
    untuned_100_train = pd.read_csv(untuned_100_train_path)


    x_column = 'Step'
    y_column1 = untuned_900_train.columns[1]
    y_column2 = untuned_100_train.columns[1]
    y_column3 = untuned_900_test.columns[1]
    y_column4 = untuned_100_test.columns[1]
    untuned_900_train = np.array(untuned_900_train[y_column1])
    untuned_100_train = np.array(untuned_100_train[y_column2])
    untuned_900_test = np.array(untuned_900_test[y_column3])
    untuned_100_test = np.array(untuned_100_test[y_column4])

    untuned_train = np.concatenate((untuned_900_train[:910], untuned_100_train[10:]), axis = 0)
    untuned_test = np.concatenate((untuned_900_test[:910], untuned_100_test[10:]), axis = 0)

    plt.figure(figsize=(10,6))
    plt.plot(untuned_train, color = 'blue', label = 'Training Loss')
    plt.plot(untuned_test, color = 'red', label = 'Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Untuned LSTM Loss vs Epoch')
    plt.legend()
    #plt.show()
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/untuned_LSTM_loss_plot.png', dpi = 300)

LSTM_loss_plots()