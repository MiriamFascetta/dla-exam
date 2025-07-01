# DLA Laboratories

## Brief summary of labs results

### Lab 1 - CNNs

#### Exercise 1.1
Write and train a basic MLP to use on MNIST dataset

'''

class BasicMLP(nn.Module):
    def __init__(self, input_size=28*28, width=64, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, output_size)

'''

'''

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

'''

ll
