import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyNetwork, self).__init__()
       
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, x):
        '''
        This function passes the data x into the model, and returns
        the final output.
        '''

        out = self.fc1(x)
    
        out = self.fc2(out)
      
        out = self.fc3(out)

        return out
