import torch.nn as nn
import torch
import ipdb

class Encoder(nn.Module):
    def __init__(self, input_size=5 ,hidden_size=256):\
        
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden_state = torch.zeros(1, self.hidden_size).to('cuda')
        self.input_size = input_size
        self.model = nn.Sequential(
                        nn.Linear(input_size + hidden_size, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, self.hidden_size)
                    )

    def reset_hidden_state(self):
        self.hidden_state = torch.zeros(1,self.hidden_size).to('cuda')

    def forward(self, input):
        ipdb.set_trace()

        input = torch.cat((input, self.hidden_state), dim=1)
        output = self.model(input)
        self.hidden_state = output
        return output


class Decoder(nn.Module):
    def __init__(self, input_size=256, output_size=5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.get_robot = nn.Sequential(
                        nn.Linear(input_size, self.input_size),
                        nn.ReLU(),
                        nn.Linear(self.input_size, self.input_size),
                        nn.ReLU(),
                        nn.Linear(self.input_size, self.output_size)
                    )
        
        self.get_hidden_state = nn.Sequential(
                        nn.Linear(input_size, self.input_size),
                        nn.ReLU(),
                        nn.Linear(self.input_size, self.input_size),
                        nn.ReLU(),
                        nn.Linear(self.input_size, self.input_size)
                    )

    def forward(self, input):
        # ipdb.set_trace()

        output = self.get_robot(input)
        hidden_state = self.get_hidden_state(input)

        return output, hidden_state

class Autoencoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=5):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size).to('cuda')
        self.decoder = Decoder(hidden_size, output_size).to('cuda')

    def forward(self, input):

        output = self.encoder(input)
        old_hidden = self.get_hidden_state()
        output, hidden_state = self.decoder(output)
        self.set_hidden_state(hidden_state)

        return output, old_hidden, hidden_state

    def reset_hidden_state(self):
        self.encoder.reset_hidden_state()

    def get_hidden_state(self):
        return self.encoder.hidden_state

    def set_hidden_state(self, hidden_state):
        self.encoder.hidden_state = hidden_state

    def get_output(self, input):

        output = self.encoder(input)
        output = self.decoder(output)
        return output

    def get_encoder_output(self, input):

        output = self.encoder(input)
        return output

    def get_decoder_output(self, input):

        output = self.decoder(input)
        return output