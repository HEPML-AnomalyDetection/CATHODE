import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml


class Classifier(nn.Module):
    def __init__(self, layers, n_inputs=5):
        super().__init__()

        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)

    def predict(self, x):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x, device=device)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


def build_classifier(filename, n_inputs=5):
    with open(filename, 'r') as stream:
        params = yaml.safe_load(stream)

    model = Classifier(params['layers'], n_inputs=n_inputs)
    if params['loss'] == 'binary_crossentropy':
        loss = F.binary_cross_entropy
    else:
        raise NotImplementedError

    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=float(params['learning_rate']))
    else:
         raise NotImplementedError       

    return model, loss, optimizer


if __name__ == '__main__':
    net = Classifier('classifier.yml', n_inputs=5)
    x = torch.rand(5)
    print(net(x))
