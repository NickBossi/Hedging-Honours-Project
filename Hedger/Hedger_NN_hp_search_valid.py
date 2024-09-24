import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold

from data_preprocessing import BS_hedge, get_sigma
import optuna
from optuna.trial import TrialState
from optuna.importance import get_param_importances
import wandb

#Setting seed
seed_number = 2024
np.random.seed(2024)
torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#Gets original paths, generated paths and the option prices of the generated paths

original_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/train_stock_data.csv"
test_path = "d:/Documents/UCT/Honours/Honours_Project/Code/data/test_stock_data.csv"

original_data = np.genfromtxt(original_path, delimiter=',', skip_header=1)
test_data = np.genfromtxt(test_path, delimiter=',', skip_header=1)
test_data = np.concatenate((original_data[-1:], test_data), axis = 0)   #Adds t=0 to the test data


generated_directory = "d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_paths.npy"
generated_paths = np.load(generated_directory)
generated_paths = np.array([path+original_data[-1] for path in generated_paths])[:,:-1,:]

option_prices = np.load("d:/Documents/UCT/Honours/Honours_Project/Code/data/H_prices.npy")
final_test_prices = np.load("d:/Documents/UCT/Honours/Honours_Project/Code/data/final_test_prices.npy")

T = generated_paths.shape[1]/365
M = 365
times = np.linspace(0, T, int(T*M))
Tts = times[-1] - times

Tts_reshaped = Tts[:generated_paths.shape[1]].reshape(1, -1, 1)

test_data = test_data.reshape(1,-1,2)
Tts_expanded = np.tile(Tts_reshaped, (test_data.shape[0], 1, 1))
test_data = np.concatenate((test_data, Tts_expanded), axis=2)

Tts_expanded = np.tile(Tts_reshaped, (generated_paths.shape[0], 1, 1))

# Concatenate time to maturity along the third dimension
data = np.concatenate((generated_paths, Tts_expanded), axis=2)

num_samples = data.shape[0]

# Define the sizes for training, validation and test data
train_size = int(num_samples * 0.7)
val_size = int(num_samples * 0.2)
test_size = num_samples - train_size - val_size  # Ensure all samples are used

# Randomly select indices for each set
all_indices = np.arange(num_samples)
np.random.shuffle(all_indices)

#Splits into test, valid and train data

train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:train_size + val_size]
test_indices = all_indices[train_size + val_size:]
non_test_indices = np.concatenate((train_indices, val_indices))

#Splits prices along same indices as the generated paths

Y = torch.tensor(option_prices[train_indices], dtype= torch.float32).to(device)
Y_val = torch.tensor(option_prices[val_indices], dtype = torch.float32).to(device)
Y_test = torch.tensor(option_prices[test_indices], dtype = torch.float32).to(device)
Y_train_final = torch.tensor(option_prices[non_test_indices], dtype = torch.float32).to(device)

# Split the data into train, validation and test sets

X = torch.tensor(data[train_indices], dtype = torch.float32).to(device)
X_val = torch.tensor(data[val_indices], dtype = torch.float32).to(device)
X_test = torch.tensor(data[test_indices], dtype = torch.float32).to(device)  
X_train_final = torch.tensor(data[non_test_indices], dtype = torch.float32).to(device)

#Final test data
final_test = torch.tensor(test_data, dtype = torch.float32).to(device)

#Gets changes in stock price and option prices
dS = X[:,1:,1:]- X[:,:-1,1:]
dHi = Y[:,1:] - Y[:,:-1]


class CustomDataset(Dataset):
    def __init__(self, path, H_values, transform = None):
        self.path = path
        self.H_values = H_values
        self.transform = transform

    def __len__(self):
        return(len(self.path))

    def __getitem__(self, index):
        path = self.path[index]
        H = self.H_values[index]
        return path, H

train_dataset = CustomDataset(X, Y)
val_dataset = CustomDataset(X_val, Y_val)
final_train_dataset = CustomDataset(X_train_final, Y_train_final)

val_loader = DataLoader(val_dataset, batch_size = 1000, shuffle = False)


best_loss = float('inf')
best_model_state = None

def define_model(input_dim = 3, num_layers = None,  activation = None, n_units = None, dropouts = None):
    layers = []
    for i in range(num_layers):
        hidden_dim = n_units[i]
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'Relu':
            layers.append(nn.ReLU())
        elif activation == 'Tanh':
            layers.append(nn.Tanh())
        elif activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(dropouts[i]))
        input_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, 2))

    return nn.Sequential(*layers)

num_epochs = 100

def objective(trial):
    torch.cuda.empty_cache()
    # Hyperparameters to search over
    num_layers = trial.suggest_int('num_layers', 1, 5)
    n_units = [trial.suggest_int(f"n_units_l{i}", 32, 512) for i in range(num_layers)]
    dropouts = [trial.suggest_float(f"dropout_l{i}", 0.0, 0.3) for i in range(num_layers)]
    activation = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'LeakyReLU'])
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 256, 1024, 4096, 8192])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    l1_reg = trial.suggest_float('l1_reg', 0, 1e-2)
    l2_reg = trial.suggest_float('l2_reg', 0, 1e-2)

    # Initialize the model
    model = define_model(num_layers=num_layers, activation=activation, n_units = n_units, dropouts = dropouts).to(device)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Regularization (L1)
    l1_lambda = l1_reg

    loss_fn = nn.MSELoss()

    #Trains for num_epochs
    for epoch in range(num_epochs):
        if epoch%10 == 0:
            print(f"Epoch {epoch} of trial {trial.number}")
        
        #Train model
        model.train()
        for i, (path, H) in enumerate(train_loader):
            path = path.to(device)
            H = H.to(device)
            holdings = model(path[:,:-1,:])
            dS = path[:,1:,1:]-path[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = H[:,1:] - H[:,:-1]

            optimizer.zero_grad()

            loss = loss_fn(dV, dHi)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

        # Evaluate model on the validation set
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for i, (path,H) in enumerate(val_loader):
                path = path.to(device)
                H = H.to(device)
                holdings = model(path[:,:-1,:])
                dS = path[:,1:,1:]-path[:,:-1,1:]
                dV = holdings*dS
                dV = torch.sum(dV, dim = 2)
                dHi = H[:,1:] - H[:,:-1]
                loss = loss_fn(dV, dHi)

                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        trial.report(valid_loss, epoch)

        if epoch>30:
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    #saves model from trial
    name = f'data/model_state_dict_{trial.number}.pth'
    torch.save(model.state_dict(), name)

    return valid_loss

study = optuna.create_study(direction="minimize",
                            study_name="final_deep_hedger",
                            storage = "sqlite:///final_deep_hedger.db",
                            load_if_exists=True)

def train_final(epochs = 1000):

    trial = study.best_trial
    trial_number = trial.number
    model_path = f'data/model_state_dict_{trial_number}.pth'
    # Retrieve the hyperparameters used in this trial
    num_layers = trial.params['num_layers']
    n_units = [trial.params[f'n_units_l{i}'] for i in range(num_layers)]
    dropouts = [trial.params[f'dropout_l{i}'] for i in range(num_layers)]
    activation = trial.params['activation']
    optimizer_name = trial.params['optimizer']
    learning_rate = trial.params['lr']
    l1_reg = trial.params['l1_reg']
    l2_reg = trial.params['l2_reg']
    batch = trial.params['batch_size']
    #batch = 100000

    wandb.login()
    run = wandb.init(project = "Final Deep Hedger",
            config = {
                "num_layers": num_layers,
                "n_units": n_units,
                "dropouts": dropouts,
                "activation": activation,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "l1_reg": l1_reg,
                "l2_reg": l2_reg,
                "batch_size": batch,
                "seed_number": seed_number
            })
    
    # Rebuild the model using the retrieved hyperparameters
    model = define_model(num_layers=num_layers, activation=activation, n_units=n_units, dropouts=dropouts).to(device)
    
    # Load the saved model state
    #model.load_state_dict(torch.load(model_path))
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_loader = DataLoader(final_train_dataset, batch_size = batch, shuffle = True)

    loss_fn = nn.MSELoss()

    training_losses = []
    test_losses = []
    #Trains for num_epochs
    for epoch in range(epochs):

        epoch_loss = 0
        
        #Train model
        model.train()
        for i, (path, H) in enumerate(train_loader):
            path = path.to(device)
            H = H.to(device)
            holdings = model(path[:,:-1,:])
            dS = path[:,1:,1:]-path[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = H[:,1:] - H[:,:-1]

            optimizer.zero_grad()

            loss = loss_fn(dV, dHi)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_reg * l1_norm
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader)

        with torch.no_grad():
            test_holdings = model(X_test[:,:-1,:])
            dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
            dV = test_holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Y_test[:,1:] - Y_test[:,:-1]
            test_loss = loss_fn(dV, dHi)
            test_losses.append(test_loss.item())
            training_losses.append(epoch_loss)
        
        print(f"Epoch {epoch} train loss: {epoch_loss}, test loss: {test_loss.item()}")
        wandb.log({"Train loss": epoch_loss, "Test loss": test_loss.item()})
        np.save('data/training_losses.npy', np.array(training_losses))
        np.save('data/test_losses.npy', np.array(test_losses))

        if epoch%100:
            torch.save(model.state_dict(), 'data/final_model_state_dict.pth')

    #Saves final model:
    torch.save(model.state_dict(), 'data/final_model_state_dict.pth')

def run_trial(num_trials):

    study.optimize(objective, n_trials = num_trials)

    importances = get_param_importances(study)
    print("Importances: ", importances)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def test_model(trial_number=None, final = False, train = False):
    # Define the path to the saved model state
    if trial_number is None:
        trial = study.best_trial
        trial_number = trial.number

    #model_path = f'data/model_state_dict_{trial_number}.pth'
    model_path = f'data/final_model_state_dict.pth'
    
    # Initialize the model with the same hyperparameters used during the trial
    trial = study.trials[trial_number]
    
    # Retrieve the hyperparameters used in this trial
    num_layers = trial.params['num_layers']
    n_units = [trial.params[f'n_units_l{i}'] for i in range(num_layers)]
    dropouts = [trial.params[f'dropout_l{i}'] for i in range(num_layers)]
    activation = trial.params['activation']
    
    # Rebuild the model using the retrieved hyperparameters
    model = define_model(num_layers=num_layers, activation=activation, n_units=n_units, dropouts=dropouts).to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(model_path))
    
    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        V0 = Y[0][0]                        #gets the price of the option at time zero
        holdings = model(X_test[:,:-1,:])
        dV = X_test[:,1:,1:]-X_test[:,:-1,1:]
        gains = holdings*dV
        gains = torch.sum(gains, dim = 1)
        VT = V0 + torch.sum(gains, dim = 1)
        PNLs = VT - Y_test[:,-1]
        PNLs = PNLs.cpu().detach().numpy()
        np.save('data/NN_test_PNLs.npy', PNLs)
        torch.save(torch.tensor(PNLs), 'data/NN_PNLs.pth')

        errors = []
        final_holdings = model(final_test[:,:-1,:])
        dv = final_test[:,1:,1:] - final_test[:,:-1,1:]
        gains = final_holdings*dv
        gains = torch.sum(gains, dim = 2).squeeze(0)
        V = Y[0][0]
        for i in range(test_data.shape[1]-1):
            V += gains[i]
            error = float((V - final_test_prices[i]).cpu().detach())
            errors.append(error)
    
        errors = np.array(errors)
        np.save('data/NN_final_errors.npy', errors)

        test = final_test[0,:,:-1].cpu().detach().numpy()
        (BS_errors, PNL, Vs) = BS_hedge(test, times, get_sigma(original_data)) 
        np.save('data/BS_final_errors.npy', BS_errors)


        # if not final and train:
        #     #computs PNLs on training data
        #     holdings = model(X_train_final[:,:-1,:])
        #     dV = X_train_final[:,1:,1:]-X_train_final[:,:-1,1:]
        #     gains = holdings*dV
        #     gains = torch.sum(gains, dim = 1)
        #     VT = V0 + torch.sum(gains, dim = 1)
        #     PNLs = VT - Y_train_final[:,-1]
        #     PNLs = PNLs.cpu().detach().numpy()
        #     torch.save(torch.tensor(PNLs), 'data/NN_PNLs_training.pth')
        #     print(np.mean(PNLs))
        #     print(np.var(PNLs))

        #     range_max = max(abs(PNLs))
        #     range_min = -range_max
        #     plt.hist(PNLs, bins = num_bins, range = (range_min, range_max), color = "green", alpha = 0.5)


        #     #Plots train PNLs under BS
        #     BS_PNLS_training = torch.load('data/BS_PNLs_training.pth')
        #     print(torch.mean(BS_PNLS_training))
        #     print(torch.var(BS_PNLS_training))
        #     BS_PNLS_training = BS_PNLS_training.cpu().detach().numpy()
        #     plt.hist(BS_PNLS_training,bins = num_bins, range = (range_min, range_max), color = "blue", alpha = 0.5)
        #     plt.show()


        #elif not final and not train:
            #computes PNLs on test data
            

        # else:
        # # Testing model with final test set (real data)
      
def get_BS_PNLS():
    
    # BS_PNLs = []
    # sigma_squared = get_sigma(original_data)
    # for path in X_test:
    #     path = path[:,:-1].cpu().detach().numpy()
    #     (PNLs, PNL, Vs) = BS_hedge(path, times, sigma_squared) #V=Y[0][0].detach().cpu().numpy())
    #     BS_PNLs.append(PNL)
    # BS_PNLs = np.array(BS_PNLs)
    # range_max = max(abs(BS_PNLs))
    # range_min = - range_max
    # torch.save(torch.tensor(BS_PNLs), 'data/BS_PNLs.pth')
    # plt.hist(BS_PNLs, bins = 50, range = (range_min, range_max), color = "orange" , alpha = 0.5)
    # plt.show()

    BS_PNLs = []
    sigma_squared = get_sigma(original_data)
    for path in X_test:
        path = path[:,:-1].cpu().detach().numpy()
        (PNLs, PNL, Vs) = BS_hedge(path, times, sigma_squared) #V=Y[0][0].detach().cpu().numpy())
        BS_PNLs.append(PNL)
        print("another PNL appended")
    BS_PNLs = np.array(BS_PNLs)
    range_max = max(abs(BS_PNLs))
    range_min = - range_max
    np.save('data/BS_test_PNLs.npy', BS_PNLs)



if __name__ == '__main__':    
    #for i in range(92):
    #    run_trial(10)
    #test_model(trial_number=None, final = False, train = False)
    #get_BS_PNLS()
    test_model()
    #train_final(1000)


































































































'''
model = DeepHedger().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss_fn = nn.MSELoss()

if train_BOOL:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (path, H) in enumerate(train_loader):
            path = path.to(device)
            H = H.to(device)
            holdings = model(path[:,:-1,:])
            dS = path[:,1:,1:]-path[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = H[:,1:] - H[:,:-1]
            optimizer.zero_grad()
            loss = loss_fn(dV, dHi)
            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
        print(f"Finished epoch {epoch+1} with loss {epoch_loss/len(train_loader)}")

    torch.save(model.state_dict(), 'model_state_dict.pth')

# Test the model
model = DeepHedger().to(device)
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()

num_bins = 50

#Tests the model to see it's distribution of PNLs
with torch.no_grad():

    V0 = Y[0][0]                        #gets the price of the option at time zero
    holdings = model(test_data[:,:-1,:])
    dV = test_data[:,1:,1:]-test_data[:,:-1,1:]
    gains = holdings*dV
    gains = torch.sum(gains, dim = 1)
    VT = V0 + torch.sum(gains, dim = 1)
    PNLs = VT - Y_test[:,-1]
    PNLs = PNLs.cpu().detach().numpy()

    range_max = max(abs(PNLs))
    range_min = -range_max
    plt.hist(PNLs, bins = num_bins, range = (range_min, range_max), color = "purple", alpha = 0.5)
    #plt.show()

#Getting PNLs for BS
BS_PNLs = []
sigma_squared = get_sigma(original_data)
for path in test_data:
    path = path[:,:-1].cpu().detach().numpy()
    (PNLs, PNL, Vs) = BS_hedge(path, times, sigma_squared, V=Y[0][0].detach().cpu().numpy())
    BS_PNLs.append(PNL)
BS_PNLs = np.array(BS_PNLs)
range_max = max(abs(BS_PNLs))
range_min = - range_max
plt.hist(BS_PNLs, bins = num_bins, range = (range_min, range_max), color = "orange" , alpha = 0.5)
plt.show()

'''

# def test_model():
#     num_layers = trial.params["num_layers"]
#     activation = trial.params["activation"]
#     n_units = []
#     dropouts = []
#     for i in range(num_layers):
#         n_units.append(trial.params[f"n_units_l{i}"])
#         dropouts.append(trial.params[f"dropout_l{i}"])
    
#     model = define_model(trial = trial, num_layers = num_layers, activation = activation, n_units = n_units, dropouts = dropouts).to(device)
#     name = f'data/model_state_dict_{trial.number}.pth'
#     model.load_state_dict(torch.load(name))
#     model.eval()
#     with torch.no_grad():
#         num_bins = 50
#         V0 = Y[0][0]                        #gets the price of the option at time zero
#         holdings = model(X_test[:,:-1,:])
#         dV = X_test[:,1:,1:]-X_test[:,:-1,1:]
#         gains = holdings*dV
#         gains = torch.sum(gains, dim = 1)
#         VT = V0 + torch.sum(gains, dim = 1)
#         PNLs = VT - Y_test[:,-1]
#         PNLs = PNLs.cpu().detach().numpy()
#         torch.save(torch.tensor(PNLs), 'data/NN_PNLs.pth')

#         range_max = max(abs(PNLs))
#         range_min = -range_max
#         plt.hist(PNLs, bins = num_bins, range = (range_min, range_max), color = "purple", alpha = 0.5)
#         plt.show()

#     #with torch.no_grad():
#     #    V_0 = Y[0][0]
#     #    final_test_deltas = 






    

