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
Y_reduced = Y[:1000]
Y_val = torch.tensor(option_prices[val_indices], dtype = torch.float32).to(device)
Y_test = torch.tensor(option_prices[test_indices], dtype = torch.float32).to(device)
Y_train_final = torch.tensor(option_prices[non_test_indices], dtype = torch.float32).to(device)

# Split the data into train, validation and test sets

X = torch.tensor(data[train_indices], dtype = torch.float32).to(device)
X_reduced = X[:1000]
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
reduced_dataset = CustomDataset(X_reduced, Y_reduced)

num_epochs = 50

    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dim, dropouts = None):
        super().__init__()

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        
        # Create a list of LSTM layers with different hidden sizes
        self.lstm_layers = nn.ModuleList()

        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True)) 
        
        self.dropout_layers = nn.ModuleList()
        for i in range(self.num_layers -1):
            self.dropout_layers.append(nn.Dropout(p=dropouts[i]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)
        
        # Dropout layer before the output
        self.dropout = nn.Dropout(p=dropouts[-1])

    def forward(self, X, hidden):
        out = X
        
        # Pass through each LSTM layer
        for i, lstm in enumerate(self.lstm_layers):
            out, h = lstm(out, hidden[i])

            if i < self.num_layers -1:
                out = self.dropout_layers[i](out)
        
        # Apply dropout to the output of the final LSTM layer
        out = self.dropout(out)

        # Output layer
        holdings = self.output_layer(out)
        
        return holdings, h
    
    def init_hidden(self, batch_size):
        # Initialize the hidden and cell states for each LSTM layer
        hidden = []
        for hidden_size in self.hidden_sizes:
            h = (torch.zeros(1, batch_size, hidden_size).to(device),
                 torch.zeros(1, batch_size, hidden_size).to(device))
            hidden.append(h)
        return hidden


def objective(trial):
    torch.cuda.empty_cache()
    # Hyperparameters to search over
    num_layers = trial.suggest_int('num_layers', 1, 5)
    n_units = [trial.suggest_int(f"n_units_l{i}", 32, 512) for i in range(num_layers)]
    dropouts = [trial.suggest_float(f"dropout_l{i}", 0.0, 0.3) for i in range(num_layers)]
    #activation = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'LeakyReLU'])
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 256, 1024])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    l1_reg = trial.suggest_float('l1_reg', 0, 1e-2)
    l2_reg = trial.suggest_float('l2_reg', 0, 1e-2)

    model = LSTM(input_size = 3, hidden_sizes = n_units, output_dim =2, dropouts =dropouts).to(device)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    loss_func = nn.MSELoss()
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1} of trial {trial.number}")

        model.train()
        for i, (paths,Hs) in enumerate(train_loader):
            paths = paths.to(device)
            Hs = Hs.to(device)

            batch_size = paths.shape[0]
            hidden = model.init_hidden(batch_size)

            holdings, _ = model(paths[:,:-1,:], hidden)
            dS = paths[:,1:,1:]-paths[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Hs[:,1:] - Hs[:,:-1]

            optimizer.zero_grad()

            loss = loss_func(dV, dHi)
            epoch_loss += loss.item()
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_reg * l1_norm

            loss.backward()

            optimizer.step()

            if (i+1) %10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
        epoch_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for i, (paths,Hs) in enumerate(val_loader):
                paths = paths.to(device)
                Hs = Hs.to(device)

                batch_size = paths.shape[0]
                hidden = model.init_hidden(batch_size)

                holdings, _ = model(paths[:,:-1,:], hidden)
                dS = paths[:,1:,1:]-paths[:,:-1,1:]
                dV = holdings*dS
                dV = torch.sum(dV, dim = 2)
                dHi = Hs[:,1:] - Hs[:,:-1]

                loss = loss_func(dV, dHi)
                valid_loss += loss.item()

            valid_loss /= len(val_loader)

        print(f"Epoch {epoch}. Train loss: {epoch_loss}. Validation Loss: {valid_loss}")

        trial.report(valid_loss, epoch)
        if trial.number > 10 and epoch >10:
            if trial.should_prune():
                raise optuna.TrialPruned()
            
    #saves model from trial
    name = f'data/LSTM_state_dict_{trial.number}.pth'
    torch.save(model.state_dict(), name)

    return valid_loss

study = optuna.create_study(direction = 'minimize',
                            study_name = "final_LSTM_hedger",
                            storage = "sqlite:///failure_LSTM_hedger.db",
                            load_if_exists = True)
    

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

#FINAL LSTM
def train_final(epochs = 1000):

    trial = study.best_trial
    trial_number = trial.number

    model_path = f'data/LSTM_state_dict_{trial_number}.pth'
    #model_path = f'data/final_model_state_dict.pth'
    
    # Initialize the model with the same hyperparameters used during the trial
    trial = study.trials[trial_number]
    
    # Retrieve the hyperparameters used in this trial
    num_layers = trial.params['num_layers']
    n_units = [trial.params[f'n_units_l{i}'] for i in range(num_layers)]
    dropouts = [trial.params[f'dropout_l{i}'] for i in range(num_layers)]
    learning_rate = trial.params['lr']
    batch_size = trial.params['batch_size']
    optimizer_name = trial.params['optimizer']
    l1_reg = trial.params['l1_reg']
    l2_reg = trial.params['l2_reg']

    wandb.login()
    run = wandb.init(project = "Final LSTM Deep Hedger",
            config = {
                "num_layers": num_layers,
                "n_units": n_units,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "l1_reg": l1_reg,
                "l2_reg": l2_reg,
                "batch_size": batch_size,
                "seed_number": seed_number
            })
    
    # Rebuild the model using the retrieved hyperparameters
    model = LSTM(input_size = 3,  hidden_sizes=n_units, output_dim = 2, dropouts=dropouts).to(device)
    
    # Load the saved model state
    #model.load_state_dict(torch.load(model_path))
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_loader = DataLoader(final_train_dataset, batch_size = batch_size, shuffle = True)

    #train_loader = DataLoader(reduced_dataset, batch_size = batch_size, shuffle = True)
    loss_fn = nn.MSELoss()

    training_losses = []
    test_losses = []
    #Trains for num_epochs
    for epoch in range(epochs):

        epoch_loss = 0
        
        #Train model
        model.train()
        for i, (paths,Hs) in enumerate(train_loader):
            paths = paths.to(device)
            Hs = Hs.to(device)

            batch_size = paths.shape[0]
            hidden = model.init_hidden(batch_size)

            holdings, _ = model(paths[:,:-1,:], hidden)
            dS = paths[:,1:,1:]-paths[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Hs[:,1:] - Hs[:,:-1]

            optimizer.zero_grad()

            loss = loss_fn(dV, dHi)
            mse_loss = loss.item()
            epoch_loss += mse_loss
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_reg * l1_norm

            loss.backward()

            optimizer.step()

            if (i+1)%10 == 0:
                print(f"Epoch {epoch+1}, Loss: {mse_loss}")
        epoch_loss /= len(train_loader)
        model.eval()

        with torch.no_grad():
            batch_size = X_test.shape[0]
            hidden = model.init_hidden(batch_size)
            test_holdings, _ = model(X_test[:,:-1,:], hidden)
            dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
            dV = test_holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Y_test[:,1:] - Y_test[:,:-1]
            test_loss = loss_fn(dV, dHi)
            test_losses.append(test_loss.item())
            training_losses.append(epoch_loss)
        
        print(f"Epoch {epoch} train loss: {epoch_loss}, test loss: {test_loss.item()}")
        wandb.log({"Train loss": epoch_loss, "Test loss": test_loss.item()})

        if epoch%100:
            torch.save(model.state_dict(), 'data/final_LSTM_model_state_dict.pth')

    #Saves final model:
    torch.save(model.state_dict(), 'data/final_LSTM_model_state_dict.pth')

def test_lstm(trial_number = None):
    if trial_number is None:
        trial = study.best_trial
        trial_number = trial.number
    #model_path = f'data/final_LSTM_model_state_dict.pth'
    model_path = f'data/LSTM_state_dict_{trial_number}.pth'
    
    # Retrieve the hyperparameters used in this trial
    num_layers = trial.params['num_layers']
    n_units = [trial.params[f'n_units_l{i}'] for i in range(num_layers)]
    dropouts = [trial.params[f'dropout_l{i}'] for i in range(num_layers)]

    model = LSTM(input_size = 3,  hidden_sizes=n_units, output_dim = 2, dropouts=dropouts).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        batch_size = X_test.shape[0]
        V0= Y[0][0]
        hidden = model.init_hidden(batch_size)
        test_holdings, _ = model(X_test[:,:-1,:], hidden)
        dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
        gains = test_holdings*dS
        gains = torch.sum(gains, dim = 1)  #???????
        VT = V0 + torch.sum(gains, dim = 1)
        PNLs = VT - Y_test[:,-1]
        PNLs = PNLs.cpu().detach().numpy()
        np.save('data/LSTM_test_PNLs.npy', PNLs)

def train_failure(epochs = 100):
    trial = study.best_trial
    trial_number = trial.number
    
    # Initialize the model with the same hyperparameters used during the trial
    trial = study.trials[trial_number]
    
    # Retrieve the hyperparameters used in this trial
    num_layers = trial.params['num_layers']
    n_units = [trial.params[f'n_units_l{i}'] for i in range(num_layers)]
    dropouts = [trial.params[f'dropout_l{i}'] for i in range(num_layers)]
    learning_rate = trial.params['lr']
    batch_size = trial.params['batch_size']
    optimizer_name = trial.params['optimizer']
    l1_reg = trial.params['l1_reg']
    l2_reg = trial.params['l2_reg']

    # wandb.login()
    # run = wandb.init(project = "Final LSTM Deep Hedger",
    #         config = {
    #             "num_layers": num_layers,
    #             "n_units": n_units,
    #             "optimizer": optimizer_name,
    #             "learning_rate": learning_rate,
    #             "l1_reg": l1_reg,
    #             "l2_reg": l2_reg,
    #             "batch_size": batch_size,
    #             "seed_number": seed_number
    #         })
    
    # Rebuild the model using the retrieved hyperparameters
    model = LSTM(input_size = 3,  hidden_sizes=n_units, output_dim = 2, dropouts=dropouts).to(device)
    
    # Load the saved model state
    #model.load_state_dict(torch.load(model_path))
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    train_loader = DataLoader(final_train_dataset, batch_size = batch_size, shuffle = True)

    #train_loader = DataLoader(reduced_dataset, batch_size = batch_size, shuffle = True)
    loss_fn = nn.MSELoss()

    training_losses = []
    test_losses = []
    #Trains for num_epochs
    for epoch in range(epochs):

        epoch_loss = 0
        
        #Train model
        model.train()
        for i, (paths,Hs) in enumerate(train_loader):
            paths = paths.to(device)
            Hs = Hs.to(device)

            batch_size = paths.shape[0]
            hidden = model.init_hidden(batch_size)

            holdings, _ = model(paths[:,:-1,:], hidden)
            dS = paths[:,1:,1:]-paths[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Hs[:,1:] - Hs[:,:-1]

            optimizer.zero_grad()

            loss = loss_fn(dV, dHi)
            mse_loss = loss.item()
            epoch_loss += mse_loss
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_reg * l1_norm

            loss.backward()

            optimizer.step()

            if (i+1)%10 == 0:
                print(f"Epoch {epoch+1}, Loss: {mse_loss}")
        epoch_loss /= len(train_loader)
        model.eval()

        with torch.no_grad():
            batch_size = X_test.shape[0]
            hidden = model.init_hidden(batch_size)
            test_holdings, _ = model(X_test[:,:-1,:], hidden)
            dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
            dV = test_holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Y_test[:,1:] - Y_test[:,:-1]
            test_loss = loss_fn(dV, dHi)
            test_losses.append(test_loss.item())
            training_losses.append(epoch_loss)
        
        print(f"Epoch {epoch} train loss: {epoch_loss}, test loss: {test_loss.item()}")
        #wandb.log({"Train loss": epoch_loss, "Test loss": test_loss.item()})

        if epoch%100:
            torch.save(model.state_dict(), 'data/failure_LSTM_model_state_dict.pth')

    #Saves final model:
    torch.save(model.state_dict(), 'data/failure_LSTM_model_state_dict.pth')

def test_failure(trial_number = None):
    if trial_number is None:
        trial = study.best_trial
        trial_number = trial.number
    model_path = f'data/failure_LSTM_model_state_dict.pth'
    #model_path = f'data/LSTM_state_dict_{trial_number}.pth'
    
    # Retrieve the hyperparameters used in this trial
    num_layers = trial.params['num_layers']
    n_units = [trial.params[f'n_units_l{i}'] for i in range(num_layers)]
    dropouts = [trial.params[f'dropout_l{i}'] for i in range(num_layers)]

    model = LSTM(input_size = 3,  hidden_sizes=n_units, output_dim = 2, dropouts=dropouts).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        batch_size = X_test.shape[0]
        V0= Y[0][0]
        hidden = model.init_hidden(batch_size)
        test_holdings, _ = model(X_test[:,:-1,:], hidden)
        dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
        gains = test_holdings*dS
        gains = torch.sum(gains, dim = 1)  #???????
        VT = V0 + torch.sum(gains, dim = 1)
        PNLs = VT - Y_test[:,-1]
        PNLs = PNLs.cpu().detach().numpy()
        np.save('data/LSTMfailure_test_PNLs.npy', PNLs)


def train_basic_lstm(epochs = 1000, load_model = False):
    wandb.init(project = "Basic LSTM Hedger")
    wandb.login()
    model = LSTM(input_size = 3, hidden_sizes = [128,128], output_dim =2, dropouts = [0,0]).to(device)

    if load_model:
        model.load_state_dict(torch.load('data/LSTMbasic_state_dict.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size = 2048, shuffle = True)

    total_steps = len(train_loader)

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (paths,Hs) in enumerate(train_loader):
            paths = paths.to(device)
            Hs = Hs.to(device)

            batch_size = paths.shape[0]
            hidden = model.init_hidden(batch_size)

            holdings, _ = model(paths[:,:-1,:], hidden)
            dS = paths[:,1:,1:]-paths[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Hs[:,1:] - Hs[:,:-1]

            optimizer.zero_grad()

            loss = loss_fn(dV, dHi)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                print(f"Epoch {epoch+1}, Step [{i+1}/{total_steps}], Loss: {loss.item()}")
        epoch_loss /= len(train_loader)

        with torch.no_grad():
            batch_size = X_test.shape[0]
            hidden = model.init_hidden(batch_size)
            test_holdings, _ = model(X_test[:,:-1,:], hidden)
            dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
            dV = test_holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Y_test[:,1:] - Y_test[:,:-1]
            test_loss = loss_fn(dV, dHi)

        wandb.log({"Train loss": epoch_loss, "Test loss": test_loss.item()})
        if (epoch %100):
            torch.save(model.state_dict(), 'data/LSTMbasic_state_dict.pth')
    torch.save(model.state_dict(), 'data/LSTMbasic_state_dict.pth')

def test_basic_lstm():  
    model = LSTM(input_size = 3, hidden_sizes = [128,128], output_dim =2, dropouts = [0,0]).to(device) 
    model.load_state_dict(torch.load('data/LSTMbasic_state_dict.pth'))
    model.eval()
    num_bins = 50
    V0 = Y[0][0]
    size = X_test.shape[0]
    with torch.no_grad():
        hidden = model.init_hidden(size)
        holdings, _ = model(X_test[:size,:-1,:], hidden)
        print(holdings.shape)
        dV = (X_test[:,1:,1:]-X_test[:,:-1,1:])#[:size,:,:]
        print(dV.shape)
        gains = torch.sum(holdings*dV, dim = -1)
        print(gains.shape)
        VT = V0 + torch.sum(gains, dim = 1)
        PNLs = VT - Y_test[:,-1]
        PNLs = PNLs.cpu().numpy()
        np.save('data/LSTMbasic_test_PNLs.npy', PNLs)
        #torch.save(torch.tensor(PNLs), 'data/NN_PNLs_training.pth')

def train_big_lstm(epochs = 1000, load_model = False):
    wandb.init(project = "Big LSTM Hedger")
    wandb.login()
    model = LSTM(input_size = 3, hidden_sizes = [128,128,128], output_dim =2, dropouts = [0,0]).to(device)

    if load_model:
        model.load_state_dict(torch.load('data/LSTMbig_state_dict.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(reduced_dataset, batch_size = 1000, shuffle = True)

    total_steps = len(train_loader)

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (paths,Hs) in enumerate(train_loader):
            paths = paths.to(device)
            Hs = Hs.to(device)

            batch_size = paths.shape[0]
            hidden = model.init_hidden(batch_size)

            holdings, _ = model(paths[:,:-1,:], hidden)
            dS = paths[:,1:,1:]-paths[:,:-1,1:]
            dV = holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Hs[:,1:] - Hs[:,:-1]

            optimizer.zero_grad()

            loss = loss_fn(dV, dHi)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                print(f"Epoch {epoch+1}, Step [{i+1}/{total_steps}], Loss: {loss.item()}")
        epoch_loss /= len(train_loader)

        with torch.no_grad():
            batch_size = X_test.shape[0]
            hidden = model.init_hidden(batch_size)
            test_holdings, _ = model(X_test[:,:-1,:], hidden)
            dS = X_test[:,1:,1:]-X_test[:,:-1,1:]
            dV = test_holdings*dS
            dV = torch.sum(dV, dim = 2)
            dHi = Y_test[:,1:] - Y_test[:,:-1]
            test_loss = loss_fn(dV, dHi)

        wandb.log({"Train loss": epoch_loss, "Test loss": test_loss.item()})
        if (epoch %100):
            torch.save(model.state_dict(), 'data/LSTMbig_state_dict.pth')
    torch.save(model.state_dict(), 'data/LSTMbig_state_dict.pth')

def test_big_lstm():  
    model = LSTM(input_size = 3, hidden_sizes = [128,128,128], output_dim =2, dropouts = [0,0]).to(device) 
    model.load_state_dict(torch.load('data/LSTMbig_state_dict.pth'))
    model.eval()
    num_bins = 50
    V0 = Y[0][0]
    size = X_test.shape[0]
    with torch.no_grad():
        hidden = model.init_hidden(size)
        holdings, _ = model(X_test[:size,:-1,:], hidden)
        print(holdings.shape)
        dV = (X_test[:,1:,1:]-X_test[:,:-1,1:])#[:size,:,:]
        print(dV.shape)
        gains = torch.sum(holdings*dV, dim = -1)
        print(gains.shape)
        VT = V0 + torch.sum(gains, dim = 1)
        PNLs = VT - Y_test[:,-1]
        PNLs = PNLs.cpu().numpy()
        np.save('data/LSTMbig_test_PNLs.npy', PNLs)
        #torch.save(torch.tensor(PNLs), 'data/NN_PNLs_training.pth')
    
if __name__ == "__main__":
    train_basic_lstm(epochs = 1000)