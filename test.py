import torch

class NetBlock(torch.nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.lin = torch.nn.Linear(inDim, outDim)
    
    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x
    
class NetworkMain(torch.nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()
        
        prevDim = in_dim
        for i, dim in enumerate(layer_dims):
            setattr(self, f'block{i}', NetBlock(prevDim, dim))
            prevDim = dim
        
        self.numLayers = len(layer_dims)
        self.outProj = torch.nn.Linear(layer_dims[-1], 10)
        
        def forward(self, x):
            for i in range(self.numLayers):
                x = getattr(self, f'block{i}')(x)
            
            return self.outProj(x)

in_dim = 512
layer_dims = [512, 1024, 256]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mn = NetworkMain(in_dim, layer_dims).to(device)