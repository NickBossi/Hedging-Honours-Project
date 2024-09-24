import numpy as np 
import matplotlib.pyplot as plt
import random


original_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/train_stock_data.csv'
test_path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/test_stock_data.csv'
original_data = np.genfromtxt(original_path, delimiter=',', skip_header=1)
test_data = np.genfromtxt(test_path, delimiter=',', skip_header=1)

generated_directory = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/generated_paths.npy'
generated_paths = np.load(generated_directory)
generated_paths = np.array([path+original_data[-1] for path in generated_paths])[:,:-1,:]

plt.figure(figsize = (20,10))

# Plot original path
#plt.plot(original_data, label='Original Path')

# # Append and plot generated paths
# for i, generated_path in enumerate(generated_paths[-100:,:,:]):
#     color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
#     while color in ['#ff0000', '#000000']:  # Ensure the color is not red or black
#         color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

#     combined_path = np.concatenate((original_data, generated_path), axis=0)
#     #plt.plot(combined_path[:, 0], combined_path[:, 1], color=color)
#     plt.plot(combined_path, linewidth = 0.5)
# final_combined = np.concatenate((original_data, test_data), axis=0)
# plt.plot(final_combined[:,1], color = 'black', linewidth = 1, label = 'Microsoft Stock Price')
# plt.plot(final_combined[:,0], color = 'brown', linewidth = 1, label = 'Apple Stock Price')


# # Add legend and show plot
# plt.legend()
# plt.title('Original and Generated Paths')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.savefig('final_plots/all_original_and_generated_paths.png', dpi = 300)
# #plt.show()
# plt.clf()

num_days_original = 3
num_generated = 1000
temp_data = original_data[-num_days_original:,:][:-1,:]

for i, generated_path in enumerate(generated_paths[0:num_generated,:,:]):
   
    combined_path = np.concatenate((temp_data, generated_path), axis=0)

    plt.plot(combined_path[:,0], linewidth = 1)
final_combined = np.concatenate((original_data[-num_days_original:,:], test_data), axis=0)
plt.plot(final_combined[:,0], color = 'black', linewidth = 2, label = 'Apple Stock Price')

plt.legend()
#plt.title('Original and Generated Paths for Apple Stock')
plt.xlabel('Time')
plt.ylabel('Price')
plt.savefig('final_plots/tail_end_original_and_generated_path_Apple_Stock.png', dpi = 300)

plt.show()
plt.clf()

plt.figure(figsize = (20,10))
for i, generated_path in enumerate(generated_paths[0:num_generated ,:,:]):
   
    combined_path = np.concatenate((temp_data, generated_path), axis=0)

    plt.plot(combined_path[:,1], linewidth = 1)
final_combined = np.concatenate((original_data[-num_days_original:,:], test_data), axis=0)
plt.plot(final_combined[:,1], color = 'black', linewidth = 2, label = 'Microsoft Stock Price')


plt.legend()
#plt.title('Original and Generated Paths for Microsoft Stock')
plt.xlabel('Time')
plt.ylabel('Price')
plt.savefig('final_plots/tail_end_original_and_generated_paths_Microsoft_Stock.png', dpi = 300)
plt.show()
plt.clf()

print("Done plotting")