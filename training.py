import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Autoencoder
from utils import generate_dataset
import ipdb

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.device = device

    def train(self, epochs=10):

        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            # if epoch % 10 == 0:
            #     self.validate()
            # ipdb.set_trace()
            self.model.train()
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                robot_loss = 0
                hidden_loss = 0
                general_loss = 0
                for i in range(7):
                    input = batch[i].to(self.device)
                    output, old_hidden, hidden_state = self.model(input)

                    loss_hidden = self.loss_fn(old_hidden, hidden_state)
                    loss = self.loss_fn(output, input)
                    loss += loss_hidden
                    general_loss += loss

                    robot_loss += loss
                    hidden_loss += loss_hidden
                    # wandb.log({'train_loss': loss})
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                wandb.log({'train_robot_loss': robot_loss / 7})
                wandb.log({'train_hidden_loss': hidden_loss / 7})
                wandb.log({'train_loss': general_loss / 7})

            self.model.reset_hidden_state()
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                robot_loss = 0
                hidden_loss = 0
                general_loss = 0
                for i in range(7):
                    # ipdb.set_trace()
                    input = batch[i].to(self.device)
                    output, old_hidden, hidden_state = self.model(input)
                    loss_hidden = self.loss_fn(old_hidden, hidden_state)
                    loss = self.loss_fn(output, input)
                    loss += loss_hidden
                    general_loss += loss

                    robot_loss += loss
                    hidden_loss += loss_hidden
                    # wandb.log({'train_loss': loss})

                wandb.log({'val_robot_loss': robot_loss / 7})
                wandb.log({'val_hidden_loss': hidden_loss / 7})
                wandb.log({'val_loss': general_loss / 7})

# wb_logger = wandb.init(project='new_representation', name='v0')

model = Autoencoder()
train_dataset = generate_dataset()
val_dataset = generate_dataset()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=7)
test_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=2)

optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# wb_logger.watch(model)
trainer = Trainer(model, train_loader, test_loader, optimizer, loss_fn, device)

trainer.train(epochs=100)