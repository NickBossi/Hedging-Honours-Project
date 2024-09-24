import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import utils
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from utils.leadlag import leadlag
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime 
import sys
import json
import seaborn as sns
import re
import wandb
import random
import scipy.stats as stats

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def sanitize_filename(filename):
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '-', filename)
    # Optionally, truncate the file name if it's too long
    return filename[:200]

# Creating custom data set
class CustomDataset(Dataset):
    def __init__(self, input_tensor, conditional_tensor, transform = None):
        self.input_tensor = input_tensor
        self.conditional_tensor = conditional_tensor
        self.transform = transform

    def __len__(self):
        return(len(self.input_tensor))

    def __getitem__(self, index):
        input_item = self.input_tensor[index]
        cond_item = self.conditional_tensor[index]
        return input_item, cond_item
    
# Data prep
#data = np.load(os.path.join('data', 'training_data.npy'))                # Loads data from Stocks folder
data = np.load('d:/Documents/UCT/Honours/Honours_Project/Code/data/training_data.npy')
scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
#print("Shape: {}".format(data.shape))
data = scaler.fit_transform(data)                                                       # Normalizes data
data = torch.tensor(np.array(data), dtype=torch.float32)                                # Converts data to tensor

input_dim = data.shape[1]              # gets dimension of inputs
num_samples = data.shape[0]
print("Input dim: {}".format(input_dim))
#print(num_samples)

input_data = data[1:]
cond_data = data[:-1]

dataset = CustomDataset(input_data, cond_data)

class Encoder(nn.Module):
    
    def __init__(self, n_latent=8, n_hidden = 50, alpha = 0.003, dropout = 0):
        super(Encoder, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim*2, n_hidden)
        self.encoder_mu = nn.Linear(n_hidden, n_latent)
        self.encoder_logvar = nn.Linear(n_hidden, n_latent)

        self.dropout = nn.Dropout(p=dropout)
            
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.N = torch.distributions.Normal(0, 1)

        self.kl = 0
    
    def forward(self, X_in, cond):
        x = torch.cat([X_in, cond], dim=1)
        x = self.dropout(x)
        x = self.lrelu(self.encoder_fc1(x))
        x = self.dropout(x)
        mu = self.lrelu(self.encoder_mu(x))
        logvar = self.lrelu(self.encoder_logvar(x))
        epsilon = self.N.sample(mu.shape).cuda()
        z = mu + torch.exp(logvar) * (epsilon)
        self.kl = -0.5*(torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim = 1))
        return z
    
class Decoder(nn.Module):
        
        def __init__(self, n_latent, n_hidden = 50, alpha = 0.03, dropout = 0):
            super(Decoder, self).__init__()
            self.decoder_fc1 = nn.Linear(n_latent+input_dim, n_hidden)
            self.decoder_fc2 = nn.Linear(n_hidden, input_dim)

            self.dropout = nn.Dropout(dropout)

            self.lrelu = nn.LeakyReLU(negative_slope=alpha)

        def forward(self, z, cond):
            x = torch.cat([z, cond], dim = 1)
            x = self.dropout(x)
            x = self.lrelu(self.decoder_fc1(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.decoder_fc2(x))
            return x


class CVAE(nn.Module):
    """ Conditional Variation Auto-Encoder (CVAE)"""

    def __init__(self, config_str):
        super(CVAE, self).__init__()
        self.config_str = config_str
        self.config = json.loads(config_str)
        self.n_latent = self.config["latent_dim"]
        self.n_hidden = self.config["n_hidden"]
        self.alpha = self.config["alpha"]
        self.dropout = self.config["alpha"]
        self.encoder = Encoder(self.n_latent, self.n_hidden, self.alpha, self.dropout)
        self.decoder = Decoder(self.n_latent, self.n_hidden, self.alpha, self.dropout)

    def forward(self, X_in, cond):
        z = self.encoder(X_in, cond)
        X_out = self.decoder(z, cond)
        return X_out
    
    def set_device(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def train_model(self):

        if self.config["wandb"]:
            wandb.login()
            run = wandb.init(project = "Final_Signature_Generation",
                    config = {
                        "window_size": self.config["window_size"],
                        "batch_size": self.config["batch_size"],
                        "depth": self.config["depth"],
                        "num_stocks": self.config["num_stocks"],
                        "num_days_training": self.config["num_days"],
                        "latent_dim": self.config["latent_dim"],
                        "epochs": self.config["n_epochs"],
                        "leadlag": self.config["leadlag"],
                        "n_hidden": self.config["n_hidden"],
                        "kl_weight": self.config["alpha"]
                    })
            run_id = run.id
            with open("d:/Documents/UCT/Honours/Honours_Project/Code/data/run_id.txt", "w") as f:
                f.write(run_id)
            # with open("../Stocks/data/run_id.txt", "w") as f:
            #     f.write(run_id)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
        self.train_load = DataLoader(dataset, batch_size = self.config["batch_size"], shuffle = True, num_workers=0)\

        running_loss = 0
        for epoch in range(self.config["n_epochs"]):

            kl_weight = min(1.0, epoch / self.config["n_epochs"])

            train_MSE_loss = 0.0
            train_kl_loss = 0.0
            train_loss = 0.0
            train_steps = 0

            self.train()

            for i, (inputs, cond) in enumerate(self.train_load):
            # Forward pass

                inputs = inputs.to(self.device)
                cond = cond.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(inputs, cond)

                MSE_loss = torch.sum((inputs-outputs)**2, dim = 1)
                #print(MSE_loss.shape)

                #MSE_loss = mse_loss(inputs, outputs)
                kl_loss = self.encoder.kl

                #loss = torch.mean((num_samples/config["batch_size"])*MSE_loss + kl_loss)
                loss = torch.mean((1-self.alpha)*MSE_loss + kl_weight * self.alpha * kl_loss)

                # Backprop
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)      # Gradient clipping to help prevent gradient blow-up
                self.optimizer.step()

                # Update loss
                train_loss += loss.item()
                train_MSE_loss += torch.mean(MSE_loss)
                train_kl_loss += torch.mean(kl_loss)
                train_steps +=1

                #first_epoch_loss.append(loss.cpu().detach().numpy())
            if self.config["wandb"]:
                wandb.log({"MSE_Loss": train_MSE_loss/train_steps, "KL_Loss": train_kl_loss/train_steps, "Epoch Loss": train_loss/train_steps})

            print('EPOCH {}. Train MSE loss: {}. Train KL loss: {}. Training loss: {}.'.format(epoch+1, train_MSE_loss/train_steps, train_kl_loss/train_steps, train_loss/train_steps))

            #if ((np.abs(train_loss/train_steps - running_loss) < 0.00005) and (epoch >500)):
                #self.config["n_epochs"] = epoch
                #self.save_model()
                #break

            running_loss = train_loss/train_steps

    #def save_model(self, path='data/cvae.pth'):
    def save_model(self, path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/cvae.pth'):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config_str': self.config_str,
        }, path)

    #def load_model(self, path='data/cvae.pth'):
    def load_model(self, path = 'd:/Documents/UCT/Honours/Honours_Project/Code/data/cvae.pth'):
        checkpoint = torch.load(path)
        self.config_str = checkpoint['config_str']
        self.config = json.loads(self.config_str)
        self.__init__(self.config_str)  # Reinitialize the model with the same config
        self.load_state_dict(checkpoint['model_state_dict'])
        self.set_device()  # Ensure that the device is set after loading

    def get_embeddings(self):
        self = self.to(self.device)
        self.latent_embeddings = []
        self.data = DataLoader(dataset, batch_size=1)

        # Gets encoding of each datapoint (both training and validation)
        for i, (input,cond) in enumerate(self.data):
            input = input.to(self.device)
            cond = cond.to(self.device)
            latent_embedding = self.encoder(input, cond).squeeze(0)
            self.latent_embeddings.append(latent_embedding.detach().cpu().numpy())

    def plot_latent(self):

        self = self.to(self.device)
        self.eval()

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
        title = sanitize_filename(self.config_str.replace(' ', '_').replace('{', '').replace('}', '').replace(':', '-').replace(',', '_').replace('"', ''))+"_"+timestamp

        if self.n_latent == 2:
            
            # Plots latent space
            x = [embedding[0] for embedding in self.latent_embeddings]
            y = [embedding[1] for embedding in self.latent_embeddings]
            plt.scatter(x, y, color = "blue", marker = "o", s = 100)
            plt.title("Latent Space")


            plt.legend(loc='upper left', title = title, fontsize='medium', shadow=True, frameon=True)
            plt.savefig('latent_plots/{}.png'.format(title))

            print('Number of latent embeddings = {}'.format(len(self.latent_embeddings)))
            plt.show()

        else:
            latent_embeddings = np.array(self.latent_embeddings)
            #stats.probplot(latent_embeddings, dist="norm", plot=plt)
            #plt.title('Q-Q Plot of Latent Space Samples')
            #plt.show()
            global_min = np.min(latent_embeddings)
            global_max = np.max(latent_embeddings)
            limit = max(global_max, np.abs(global_min))

            fig, axes = plt.subplots(self.n_latent, 1, figsize=(10, 2*self.n_latent))

            for i in range(self.n_latent):
                ax = axes[i]
                sns.histplot(latent_embeddings[:, i], kde=True, ax=ax)
                ax.set_title(f'Dimension {i+1}', fontsize = 10)

                ax.set_xlim(-limit, limit)
            
            #fig.legend(loc='upper left', title = title, fontsize='medium', shadow=True, frameon=True)
            
            plt.subplots_adjust(hspace=0.5)  
            plt.tight_layout(rect = [0, 0, 1, 0.95])
            plt.savefig('latent_plots/{}.png'.format(title))
            #plt.savefig('latent_plots/final_latent_plot.png')
            if self.config["show_plot"]:
                plt.show()

            #if self.config["wandb"]:
                #wandb.log({"laten_plot": fig})

    def generate_signatures(self):
        self.num_paths = self.config["num_paths"]
        self.window_size = self.config["window_size"]
        self.forecast_horizon = self.config["forecast_horizon"]

        self.latent_mean = np.mean(self.latent_embeddings, axis = 0)
        self.latent_std = np.std(self.latent_embeddings, axis = 0)

        sigs = []

        self.eval()

        for j in range(self.num_paths):
            current_path_of_sigs = []           #stores signatures generated in this current path

            #getting the last path in our data on which we will condition our first signature
            cond = (data[-1].unsqueeze(0)).to(self.device)

            for i in range(int(self.forecast_horizon/(self.window_size))+1):
                #z = torch.randn(1, self.config["latent_dim"]).to(self.device)
                sigma = np.diag(self.latent_std**2)
                z = (torch.tensor((np.random.multivariate_normal(self.latent_mean, sigma, 1)))).float().to(self.device)
                generated_sig = self.decoder(z, cond)
                current_path_of_sigs.append(generated_sig.cpu().detach().numpy().squeeze(0))
                cond = generated_sig.clone()            #sets the current signature as the condition for the next signature

            sigs.append(np.array(current_path_of_sigs))   #appends the current path to the list of all paths

        sigs = np.array(sigs)
        n = sigs.shape[0]
        m = sigs.shape[1]
        k = sigs.shape[2]
        sigs = sigs.reshape(n*m, k)
        sigs = scaler.inverse_transform(sigs)                   #undoes normalisation
        sigs = sigs.reshape(n, m, k)
        np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_logsigs.npy', sigs)
        # with open('data/generated_logsigs.npy', 'wb') as f:
        #     pickle.dump(sigs, f)

    def generate_test_signatures(self):
        self = self.to(self.device)
        #self.discriminator = self.config["process_discriminator"]
        self.window_size = self.config["window_size"]
        self.forecast_horizon = 20

        self.latent_mean = np.mean(self.latent_embeddings, axis = 0)
        self.latent_std = np.std(self.latent_embeddings, axis = 0)

        sigs = []

        self.eval()

        sigma = np.diag(self.latent_std**2)

        for i, (inputs, cond) in enumerate(self.data):
            cond = cond.to(self.device)
            #z = torch.randn(1, self.config["latent_dim"]).to(self.device)

            z = (torch.tensor((np.random.multivariate_normal(self.latent_mean, sigma, 1)))).float().to(self.device)
            generated_sig = self.decoder(z, cond)
            sigs.append(generated_sig.cpu().detach().numpy())

        sigs = np.array(sigs)
        n = sigs.shape[0]
        m = sigs.shape[1]
        k = sigs.shape[2]
        sigs = sigs.reshape(n*m, k)
        sigs = scaler.inverse_transform(sigs)                   #undoes normalisation
        sigs = sigs.reshape(n, m, k)
        np.save('d:/Documents/UCT/Honours/Honours_Project/Code/data/generated_test_logsigs.npy', sigs)
        # with open('data/generated_test_logsigs.npy', 'wb') as f:
        #     pickle.dump(sigs, f)
        print("GENERATED TEST SIGNATURES")

if __name__ == "__main__":

    config_str = sys.argv[1]
    config = json.loads(config_str)
    model = CVAE(config_str)
    model.set_device()
    if config["train_model"]:
        print(model.device)
        model.train_model()
        model.save_model()

    else:
        model.load_model()
        model.config_str = config_str
        model.config = config
        model.set_device()

    model.get_embeddings()

    if config["plot_latent"]:  
        model.plot_latent()

    if config["generate_paths"]:
        model.generate_test_signatures()
        model.generate_signatures()

    #if config["process_discriminator"]:
    #model.generate_test_signatures()

    print("Generation complete.")
    #plt.plot(first_epoch_loss)
    #plt.show()
