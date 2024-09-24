import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from scipy import stats
import random

random.seed(2024)

original_data = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/training_data.npy')[1:,:]
generated_data = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_test_logsigs.npy', allow_pickle=True)
generated_data = np.squeeze(generated_data)

# File paths for training and testing sets
train_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/train_stock_data.csv"

# Gets training and testing data
original_time_series = pd.read_csv(train_path)    #shape = (9610,2)
original_time_series = np.array(original_time_series)
generated_time_series = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/np_generated_time_series.npy')

def tsne_plot(original_data, generated_data, perplexity = 4, n_iter= 10000, random_state = 42 ):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    X_embedded = tsne.fit_transform(original_data)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], label='Original Data')
    X_embedded = tsne.fit_transform(generated_data)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], label='Generated Data')
    plt.legend()
    plt.show()
    plt.clf()

def plots():
    plt.figure(figsize=(12, 6))
    new_time_series = generated_time_series[0,1:,:]
    new_time_series = new_time_series + original_time_series[0,:]
    for i in range(generated_time_series.shape[0]):
        latest = (generated_time_series[i,:,:] +new_time_series[-1,:])[1:,:]
        new_time_series = np.concatenate((new_time_series, latest), axis=0)
    print(new_time_series.shape)

    plt.plot(original_time_series[:,1], color='orange', label = 'Original Microsoft Data')
    plt.plot(new_time_series[:,1], color='purple', label = 'Generated Microsoft Data')
    plt.plot(original_time_series[:,0], color='blue', label = 'Original Apple Data')
    plt.plot(new_time_series[:,0], color='green', label = 'Generated Apple Data')

    plt.legend()
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/Original_vs_Generated.png', dpi=300)
    #plt.show()
    original_gains_1 = []
    original_gains_2 = []
    new_gains_1 = []
    new_gains_2 = []

    window_size = 5
    for i in range(int(original_time_series.shape[0]/window_size)):
        current1 = original_time_series[i*window_size:(i+1)*window_size,:]-original_time_series[i*window_size,:]
        gain1 = current1[-1,:]
        original_gains_1.append(gain1[0])
        original_gains_2.append(gain1[1])
        current2 = new_time_series[i*window_size:(i+1)*window_size,:]-new_time_series[i*window_size,:]
        gain2 = current2[-1,:]
        new_gains_1.append(gain2[0])
        new_gains_2.append(gain2[1])
        plt.plot(current1, color='blue')
        plt.plot(current2, color='red')
    #plt.show()
    plt.clf()

    print(np.mean(original_gains_1), np.mean(new_gains_1))
    print(np.mean(original_gains_2), np.mean(new_gains_2))
    num_bins = 25
    # plt.hist(original_gains_1, bins = num_bins, color = 'blue', alpha = 0.5)
    # plt.hist(new_gains_1, bins=num_bins, color = 'red', alpha = 0.5)
    # plt.show()

    # plt.hist(original_gains_2, bins = num_bins, color = 'blue', alpha = 0.5)
    # plt.hist(new_gains_2, bins = num_bins, color = 'red', alpha = 0.5)
    # plt.show()

    plt.figure(figsize=(10, 6))


    bin_edges = np.histogram_bin_edges(np.concatenate((original_gains_1, new_gains_1)), bins=num_bins)

    # Create double histogram for original_gains_1 and new_gains_1
    plt.figure(figsize=(10, 6))
    plt.hist(original_gains_1, bins=bin_edges, color='blue', alpha=1, label='Original Gains', align='mid', width=0.4)
    plt.hist(new_gains_1, bins=bin_edges, color='red', alpha=1, label='Generated Gains', align='mid', width=0.4, rwidth=0.8)
    plt.legend()
    plt.title('Histogram of Original and Generated Gains for Apple Stock')
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/apple_gains.png', dpi=300)
    #plt.show()
    plt.clf()

    # Calculate bin edges
    bin_edges = np.histogram_bin_edges(np.concatenate((original_gains_2, new_gains_2)), bins=num_bins)

    # Create double histogram for original_gains_2 and new_gains_2
    plt.figure(figsize=(10, 6))
    plt.hist(original_gains_2, bins=bin_edges, color='blue', alpha=1, label='Original Gains', align='mid', width=0.4)
    plt.hist(new_gains_2, bins=bin_edges, color='red', alpha=1, label='Generated Gains', align='mid', width=0.4, rwidth=0.8)
    plt.legend()
    plt.title('Histogram of Original Generated Gains for Microsoft Stock')
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/microsoft_gains.png', dpi=300)
    #plt.show()
    plt.clf()


    plt.style.use('seaborn-v0_8-deep')
    #plt.style.use('Solarize_Light2')

    bins = np.histogram_bin_edges(np.concatenate((original_gains_2, new_gains_2)), bins=num_bins)
    #bins = np.linspace(-20, 20, 30)

    plt.hist([original_gains_1, new_gains_1], bins, label=['Original Gains', 'Generated Gains'])
    plt.legend(loc='upper right')
    plt.title('Histogram of Original and Generated Gains for Apple Stock')
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/apple_gains_2.png', dpi=300)
    #plt.show()
    plt.clf()

    plt.style.use('seaborn-v0_8-deep')
    #plt.style.use('Solarize_Light2')


    #bins = np.linspace(-20, 20, 30)

    plt.hist([original_gains_2, new_gains_2], bins, color = ['orange', 'purple'], label=['Original Gains', 'Generated Gains'])
    plt.legend(loc='upper right')
    plt.title('Histogram of Original and Generated Gains for Microsoft Stock')
    plt.savefig('d:/Documents/UCT/Honours/Honours_Project/plots/microsoft_gains_2.png', dpi=300)
    #plt.show()

def stat_test():
    original_gains_1 = []
    original_gains_2 = []
    new_gains_1 = []
    new_gains_2 = []

    window_size = 5

    new_time_series = generated_time_series[0,1:,:]
    new_time_series = new_time_series + original_time_series[0,:]
    for i in range(generated_time_series.shape[0]):
        latest = (generated_time_series[i,:,:] +new_time_series[-1,:])[1:,:]
        new_time_series = np.concatenate((new_time_series, latest), axis=0)


    for i in range(int(original_time_series.shape[0]/window_size)):
        current1 = original_time_series[i*window_size:(i+1)*window_size,:]-original_time_series[i*window_size,:]
        gain1 = current1[-1,:]
        original_gains_1.append(gain1[0])
        original_gains_2.append(gain1[1])
        current2 = new_time_series[i*window_size:(i+1)*window_size,:]-new_time_series[i*window_size,:]
        gain2 = current2[-1,:]
        new_gains_1.append(gain2[0])
        new_gains_2.append(gain2[1])

    ks_statistic, p_value = stats.ks_2samp(original_gains_1, new_gains_1)

    print(f"K-S Test statistic: {ks_statistic}, p-value: {p_value}")    

    ks_statistic, p_value = stats.ks_2samp(original_gains_2, new_gains_2)

    print(f"K-S Test statistic: {ks_statistic}, p-value: {p_value}")  

def conditional_test():
    original_windows = []
    generated_windows = []
    generated_MSEs = []
    true_MSEs = []
    generated_MSEs1 = []
    generated_MSEs2 = []
    true_MSEs1 = []
    true_MSEs2 = []

    window_size = 5

    new_time_series = generated_time_series[0,1:,:]
    new_time_series = new_time_series + original_time_series[0,:]
    for i in range(generated_time_series.shape[0]):
        latest = (generated_time_series[i,:,:] +new_time_series[-1,:])[1:,:]
        new_time_series = np.concatenate((new_time_series, latest), axis=0)


    for i in range(int(original_time_series.shape[0]/window_size)):
        original_window = original_time_series[i*window_size:(i+1)*window_size,:]-original_time_series[i*window_size,:]
        original_windows.append(original_window)
        generated_window = new_time_series[i*window_size:(i+1)*window_size,:]-new_time_series[i*window_size,:]
        generated_windows.append(generated_window) 
    original_windows = np.array(original_windows)
    generated_windows = np.array(generated_windows)
    print("Conditional Test Completed")

    for i in range (generated_windows.shape[0]):
        random_number = random.randint(0,generated_windows.shape[0]-1)

        MSE_true_generated = np.mean((original_windows[i] - generated_windows[random_number])**2)
        MSE_true_generated1 = np.mean((original_windows[i][:,0] - generated_windows[random_number][:,0])**2)
        MSE_true_generated2 = np.mean((original_windows[i][:,1] - generated_windows[random_number][:,1])**2)

        generated_MSEs.append(MSE_true_generated)
        generated_MSEs1.append(MSE_true_generated1)
        generated_MSEs2.append(MSE_true_generated2)

        MSE_true_true = np.mean((original_windows[i] - original_windows[random_number])**2)
        MSE_true_true1 = np.mean((original_windows[i][:,0] - original_windows[random_number][:,0])**2)
        MSE_true_true2 = np.mean((original_windows[i][:,1] - original_windows[random_number][:,1])**2)

        true_MSEs.append(MSE_true_true)
        true_MSEs1.append(MSE_true_true1)
        true_MSEs2.append(MSE_true_true2)

    print("MSEs Calculated")
    print(f"Generated MSEs: {np.mean(generated_MSEs)}\n True MSEs: {np.mean(true_MSEs)}")
    print(f"Generated MSEs1: {np.mean(generated_MSEs1)}\n True MSEs1: {np.mean(true_MSEs1)}")
    print(f"Generated MSEs2: {np.mean(generated_MSEs2)}\n True MSEs2: {np.mean(true_MSEs2)}")


conditional_test()
