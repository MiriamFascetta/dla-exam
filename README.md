# Brief summary of labs results

## Lab 1 - CNNs

### Exercise 1.1 - Basic MLP
Write and train a basic MLP to use on MNIST dataset
Network:

```
class BasicMLP(nn.Module):
    def __init__(self, input_size=28*28, width=64, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
```

Training with SGD optimizer:
epochs = 50
lr = 0.1
plots

Training with Adam optimizer:
epochs = 50
lr = 0.001
plots

From now on, Adam optimizer was used if not specified.


### Exercise 1.2: MLP depth experiments with and without residual connections

Network:

```
class LinearBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return F.relu(self.fc(x))


class ResMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(LinearBlock(width), LinearBlock(width))

    def forward(self, x):
        return self.network(x) + x


class Net(nn.Module):
    def __init__(self, input_size=28*28, width=64, depth=1, residual=False, output_size=10):

        super().__init__()

        blocks = [ResMLPBlock(width) if residual else LinearBlock(width) for _ in range(depth)]
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, width),
            nn.ReLU(),
            *blocks,
            nn.Linear(width, output_size)
        )

    def forward(self, x):
        return self.network(x)
```

Training without residual connections:
epochs = 50
lr = 0.001
width = 64

depth = 10:
plots

depth = 20:

depth = 30:

Gradients of the last model:
plot


Training with residual connections:
epochs = 50
lr = 0.001
width = 64

depth = 5:
plots

depth = 10:

depth = 15:

Gradients of the last model:
plot

We can clearly see that even deep networks are able to learn because of the gradient signal


### Exercise 1.3: CNNs trained on Cifar10 with and without residual connections

Network:

```
class LinearBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return F.relu(self.fc(x))


class ResMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(LinearBlock(width), LinearBlock(width))

    def forward(self, x):
        return self.network(x) + x


class BasicConvBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.residual:
            out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, mode='mlp', input_size=28*28, in_channels=1, width=64, base_channels=64,
                 depth=1, residual=False, output_size=10):
        super().__init__()
        self.mode = mode

        if mode == 'mlp':
            blocks = [ResMLPBlock(width) if residual else LinearBlock(width) for _ in range(depth)]
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, width),
                nn.ReLU(),
                *blocks,
                nn.Linear(width, output_size)
            )
        elif mode == 'cnn':
            blocks = [BasicConvBlock(base_channels, base_channels, residual=residual) for _ in range(depth)]
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                *blocks,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(base_channels, output_size)
            )
        else:
            raise ValueError("Mode must be either 'mlp' or 'cnn'")

    def forward(self, x):
        return self.network(x)
```

Training with residual connections:
epochs = 100
lr = 0.001
base_channels=64

depth = 5:
plots

depth = 10:

depth = 15:

Comment:
The accuracy increases with the depth of the network


Training without residual connections:
epochs = 100
lr = 0.001
base_channels=64

depth = 5:
plots

depth = 10:

depth = 15:

Comment:
The accuracy decreases with the depth of the network


### Exercise 2.1: Fine-tuning of a CNN pre-trained on Cifar10 using Cifar100

Network modified to support feature extraction:

```
class Net(nn.Module):
    def __init__(self, mode='mlp', input_size=28*28, in_channels=1, width=64, base_channels=64,
                 depth=1, residual=False, output_size=10, return_features=False):
        super().__init__()
        self.mode = mode
        self.return_features = return_features

        if mode == 'mlp':
            blocks = [ResMLPBlock(width) if residual else LinearBlock(width) for _ in range(depth)]
            self.feature_extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, width),
                nn.ReLU(),
                *blocks
            )
            self.classifier = nn.Linear(width, output_size)

        elif mode == 'cnn':
            blocks = [BasicConvBlock(base_channels, base_channels, residual=residual) for _ in range(depth)]
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                *blocks,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.fc = nn.Linear(base_channels, base_channels)
            self.classifier = nn.Linear(base_channels, output_size)
        else:
            raise ValueError("Mode must be either 'mlp' or 'cnn'")

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.return_features:
            return features
        return self.classifier(features)

    def get_features(self, x):
        return self.feature_extractor(x)
```

CNN of depth = 5 was used as feature extractor.
Initially classical classifiers were trained to get a baseline
LinearSVM accuracy on validation data: 30.24%
K-Nearest Neighbor accuracyon validation data: 16.11%

Then, the final classifier was replaced to output 100, and the fc and classifier layers parameter were finetuned reaching the baseline accuracy of 30.63% on validation data.

Then all the parameters were finetuned reaching an accuracy of 55.48% using Adam optimizer with lr=0.001 and of 532.67% using SGD with lr=0.01.

A model was trained on Cifar100 to compare the results reaching an accuracy of 55.41%.
















