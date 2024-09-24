import torch
import numpy as np
import matplotlib.pyplot as plt

NN_PNLs = np.load('data/NN_test_PNLs.npy')
LSTM_PNLs = np.load("data/LSTM_test_PNLs.npy")
LSTM_basic_PNLS = np.load('data/LSTMbasic_test_PNLs.npy')
LSTM_big_PNLs = np.load("data/LSTMbig_test_PNLs.npy")
LSTM_failure_PNLs = np.load("data/LSTMfailure_test_PNLs.npy")
BS_PNLs = np.load("data/BS_test_PNLs.npy")
BS_final_PNLs = np.load("data/BS_final_errors.npy")
NN_final_PNLs = np.load("data/NN_final_errors.npy")

range1 = max(np.max(NN_PNLs), np.abs(np.min(NN_PNLs)))
range2 = max(np.max(LSTM_PNLs), np.abs(np.min(LSTM_PNLs)))
range3 = max(np.max(BS_PNLs), np.abs(np.min(BS_PNLs)))

final_range = max(range1, range2, range3)


print(f"NN PNLS: {np.mean(NN_PNLs)}")
print(f"NN Variance: {np.var(NN_PNLs)}")
print(f"LSTM PNLS: {np.mean(LSTM_PNLs)}")
print(f"LSTM Variance: {np.var(LSTM_PNLs)}")
print(f"BS PNLS: {np.mean(BS_PNLs)}")
print(f"BS Variance: {np.var(BS_PNLs)}")
print(f"LSTM Basic PNLS: {np.mean(LSTM_basic_PNLS)}")
print(f"LSTM Basic Variance: {np.var(LSTM_basic_PNLS)}")
print(f"LSTM Big PNLS: {np.mean(LSTM_big_PNLs)}")
print(f"LSTM Big Variance: {np.var(LSTM_big_PNLs)}")
print(f"LSTM Failure PNLS: {np.mean(LSTM_failure_PNLs)}")
print(f"LSTM Failure Variance: {np.var(LSTM_failure_PNLs)}")


num_bins = 100
plt.hist(LSTM_PNLs, bins = num_bins, color = "purple", range = (-final_range, final_range), alpha = 0.5, label = "LSTM")
plt.hist(NN_PNLs, bins = num_bins, color = "orange", range = (-final_range, final_range), alpha = 0.5, label = "NN")
plt.hist(BS_PNLs, bins = num_bins, color = "blue", range = (-final_range, final_range), alpha = 0.5, label = "BS")
plt.hist(LSTM_basic_PNLS, bins = num_bins, color = "green", range = (-final_range, final_range), alpha = 0.5, label = "LSTM Basic")

plt.savefig(r"d:\Documents\UCT\Honours\Honours_Project\plots")
#plt.show()
