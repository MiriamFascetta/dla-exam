# Brief summary of labs results

## Lab 1 - CNNs

### Exercise 1.1 - Basic MLP on MNIST
Implement and train a simple Multi-Layer Perceptron on the MNIST dataset.

**Model architecture**:

```ruby
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

**Training setup**:
- SGD: 50 epochs, learning rate = 0.1
- Adam: 50 epochs, learning rate = 0.001

Plots for SGD
![Screenshot 2025-07-02 120015](https://github.com/user-attachments/assets/1e4637b0-acc5-45a2-bc37-881fe41d883e)

Plots for Adam
![Screenshot 2025-07-02 120256](https://github.com/user-attachments/assets/f79d7607-aa33-45e2-aa07-358d1cb9b842)

Adam is used as the default optimizer for all following experiments unless otherwise specified.


### Exercise 1.2: MLP depth experiments with and without residual connections

**Model architecture**:

```ruby
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

Training setup (no residuals):
- 50 epochs, lr = 0.001, width = 64
- Tested depths: 10, 20, 30

Plots for depth = 10:
![Screenshot 2025-07-02 120353](https://github.com/user-attachments/assets/5229510e-29f6-4dc3-b895-e3a4e7fe8962)

Plots for depth = 20:
![Screenshot 2025-07-02 120422](https://github.com/user-attachments/assets/fd66413d-9042-402a-bf20-d38282417b7d)

Plots for depth = 30:
![Screenshot 2025-07-02 120505](https://github.com/user-attachments/assets/7362c784-7a02-4234-b972-8eea4b81ed63)

Gradient plots of the last model:
![Screenshot 2025-07-02 120534](https://github.com/user-attachments/assets/8e6f4e7c-9a15-407c-95ac-8abf696267d3)


**Training setup (with residuals)**:
- 50 epochs, lr = 0.001, width = 64
- Tested depths: 5, 10, 15

Plots for depth = 5:
![Screenshot 2025-07-02 120629](https://github.com/user-attachments/assets/1c6595fc-dc41-411d-81ee-e877f6c39b97)

Plots for depth = 10:
![Screenshot 2025-07-02 120656](https://github.com/user-attachments/assets/8ea7347a-19d1-4383-926b-3d2cc9c80f02)

Plots for depth = 15:
![Screenshot 2025-07-02 120719](https://github.com/user-attachments/assets/ee0cff2c-6e32-4623-8157-db4f11b850b3)

Gradient plots of the last model:
![Screenshot 2025-07-02 120802](https://github.com/user-attachments/assets/bea5f00e-aa67-4297-9b34-5d32d074c387)


**Comment**:
Without residual connections, deeper networks suffered from vanishing gradients and struggled to converge.
With residual connections, even deeper MLPs learned effectively, as shown by the gradient signal and improved training performance.


### Exercise 1.3: CNNs trained on Cifar10 with and without residual connections

**Model architecture**:

```ruby
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

**Training setup (with residuals)**:
- 100 epochs, lr = 0.001, base_channels = 64
- Depths tested: 5, 10, 15

Plots for depth = 5:
![Screenshot 2025-07-02 120901](https://github.com/user-attachments/assets/5afdc94d-c8e9-491b-a872-a48c81b1add9)

Plots for depth = 10:
![Screenshot 2025-07-02 120931](https://github.com/user-attachments/assets/486cfcb1-dcba-416f-b971-d19bcfb03f71)

Plots for depth = 15:
![Screenshot 2025-07-02 120957](https://github.com/user-attachments/assets/f781e484-c29f-4a37-94c7-6a0d6f1caff4)


**Comment**:
Accuracy consistently increased with depth, thanks to residual connections which preserved gradient signal.


Training setup (without residuals):
- 100 epochs, lr = 0.001, base_channels = 64
- Depths tested: 5, 10, 15

Plots for depth = 5:
![Screenshot 2025-07-02 121057](https://github.com/user-attachments/assets/d6f5531e-9219-463f-b19c-ad7137af886d)

Plots for depth = 10:
![Screenshot 2025-07-02 121126](https://github.com/user-attachments/assets/8f34943e-9fbd-4b4b-b6fa-d920b97401e4)

Plots for depth = 15:
![Screenshot 2025-07-02 121149](https://github.com/user-attachments/assets/e8a975dd-3964-4a13-b40d-903472a49715)


**Comment**:
Accuracy decreased with network depth due to training difficulties like vanishing gradients from missing residual connections.


### Exercise 2.1: Fine-tuning CIFAR10 CNN on CIFAR100

**Updated model architecture**:

```ruby
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

**Setup**:
- Feature extractor: CNN (depth = 5) pretrained on CIFAR10

**Classical classifiers trained on extracted features**:
- Linear SVM: 30.24% validation accuracy
- K-NN: 16.11% validation accuracy

**Fine-tuning results**:
- fc & classifier layers only: 30.63%
- Full network (Adam, lr=0.001): 55.48%
- Full network (SGD, lr=0.01): 52.67%

**Training from scratch on CIFAR100**: 55.41%

Plots for fc & classifier layers only
![Screenshot 2025-07-02 121358](https://github.com/user-attachments/assets/78714cec-2d5b-4a90-a868-c89c7f62e3c4)

Plots for full network using Adam
![Screenshot 2025-07-02 121501](https://github.com/user-attachments/assets/a1e734ba-2125-4836-bdcf-5e31d3b78f16)

Plots for full network using SGD
![Screenshot 2025-07-02 121540](https://github.com/user-attachments/assets/4dc52725-748e-42b9-8521-2ff97234d832)

Plots for training from scratch on CIFAR100
![Screenshot 2025-07-02 121636](https://github.com/user-attachments/assets/a8a117cd-e20b-4349-921e-21119b5b370d)


**Comment**:
Fine-tuning the full network significantly outperformed classical classifiers like SVM and K-NN. However, training from scratch on CIFAR100 achieved nearly identical performance, suggesting that the dataset size and model capacity may have been too limited for fine-tuning to offer a significant advantage.



## Lab 3 - Transformers

This lab focuses on using DistilBERT for text classification on the Rotten Tomatoes sentiment dataset.


### Exercises 1.1, 1.2 and 1.3.1

The `[CLS]` token from the last transformer layer of the DistilBERT base (uncased) model was used as a sentence-level feature representation.
The `[CLS]` embedding captures global semantic information from each sentence, making it a suitable input for downstream classifiers. 


### Exercises 1.3.2 and 1.3.3

Classical classifiers were trained using the `[CLS]` features:
- Support Vector Machine (SVM): 82% validation accuracy
- Logistic Regression: 83% validation accuracy
- K-Nearest Neighbors (K-NN): 74% validation accuracy


### Exercises 2.1, 2.2, 2.3 and 3.1

A classification head was added on top of the `[CLS]` token, and the model was fine-tuned for sentiment classification.
- Baseline (No fine-tuning): Accuracy: 50%
  The pretrained model without task-specific fine-tuning behaves as a random classifier. This is expected because the newly added classification head is randomly initialized.

Several fine-tuning strategies were explored for DistilBERT. Some approaches were computationally intensive due to the large size of DistilBERT, while others offered substantial memory and time savings without sacrificing accuracy:
- Full Fine-Tuning (32-bit):
  * Accuracy: 83.96%
  * Time: 8:46 (3 epochs)
- Fine-Tune Classifier + Last Transformer Layer:
  * Accuracy: 85.27%
  * Time: 8:59 (3 epochs)
- Full Fine-Tuning with Mixed Precision (16-bit):
  * Accuracy: 85.08%
  * Time: 3:30 (3 epochs)
- Parameter-Efficient Fine-Tuning (LoRA):
  * Accuracy: 83.20%
  * Time: 7:12 (3 epochs)
- Using DistilBERT Pretrained on SST-2:
  * Accuracy: 90.53%
  * Time: 0:11



## Lab 4 - Adversarial Learning and OOD Detection

### Exercises 1.1 and 1.2 

To build a basic OOD detection pipeline:
- **In-Distribution (ID)**: CIFAR-10
- **Out-of-Distribution (OOD)**: Subset of 20 non-overlapping classes from CIFAR-100
- **Model**: CNN of depth 5 (trained in Lab 1)
- **Detection Tools**:
  - Histograms of logits and softmax probabilities
  - ROC & Precision-Recall curves
  - Autoencoder loss (decoder trained, encoder reused from CNN)

![Screenshot 2025-07-02 122816](https://github.com/user-attachments/assets/5c4fb3c9-6898-45ac-8c73-3b8b7ac4f72f)

![Screenshot 2025-07-02 122910](https://github.com/user-attachments/assets/e7fed3fa-01be-46d1-97a6-b4829a132e5c)

![Screenshot 2025-07-02 160523](https://github.com/user-attachments/assets/cb27d600-78c5-4beb-902b-c462bfdf088a)

![Screenshot 2025-07-02 160716](https://github.com/user-attachments/assets/e0eea19d-2736-490d-904c-6e556050fcec)

![Screenshot 2025-07-02 160745](https://github.com/user-attachments/assets/8379cbe9-c2f5-4efb-a4eb-5a8c1f554393)

![Screenshot 2025-07-02 122955](https://github.com/user-attachments/assets/2b113390-6609-45bb-a3f3-ee470d74915c)


### Exercise 2.1 

Adversarial examples were generated using Fast Gradient Sign Method (FGSM) and test examples from CIFAR-10 for various ε values:

- ε: 0.00 (no pertubation), Accuracy on adversarial: 84.60%
- ε: 0.01, Accuracy on adversarial: 47.47%
- ε: 0.05, Accuracy on adversarial: 10.18%
- ε: 0.10, Accuracy on adversarial: 4.85%
- ε: 0.20, Accuracy on adversarial: 4.38%
- ε: 0.30, Accuracy on adversarial: 5.41%

**Comment**:
As ε increases, the perturbation becomes more aggressive, leading to a drop in accuracy.


### Exercise 2.2

To increase robustness, FGSM is used to generate adversarial samples during training. The following composite loss was used:

L(θ, x, y)= α * L(θ, x, y) + (1 - α) * L(θ, x + ε * sign (∇ₓ L(θ, x, y_TRUE), y))

with ε = 0.01.

Then the model is evaluated using the OOD detection setup and metrics from Exercise 1.

The accuracies after training:

- ε: 0.00 (no pertubation), Accuracy on adversarial: 86.81%
- ε: 0.01, Accuracy on adversarial: 76.38%
- ε: 0.05, Accuracy on adversarial: 30.00%
- ε: 0.10, Accuracy on adversarial: 9.39%
- ε: 0.20, Accuracy on adversarial: 3.26%
- ε: 0.30, Accuracy on adversarial: 2.08%

![Screenshot 2025-07-02 140138](https://github.com/user-attachments/assets/71fe3173-b199-47b0-8df3-6fd738c534d7)

![Screenshot 2025-07-02 140223](https://github.com/user-attachments/assets/71025b02-e9f5-4c49-8507-64e6a91f21f8)

![Screenshot 2025-07-02 140253](https://github.com/user-attachments/assets/8092d9a4-1518-4039-8ff3-b734c821e783)

![Screenshot 2025-07-02 140312](https://github.com/user-attachments/assets/bbeacc62-db2a-41e6-a8ac-c2aa91a8de70)

![Screenshot 2025-07-02 140333](https://github.com/user-attachments/assets/bd4e0a39-47f6-4e99-8aa4-a235ce8b7e6e)


**Comment**:
Adversarial training with ε = 0.01 significantly improved robustness at that level of attack.


### Exercise 3.1

**ODIN** was implemented to improve OOD detection using temperature scaling and input perturbation. A grid search was conducted on temperature T and ε.

Best hyperparameters:
- Temperature T = 50
- ε = 0

![Screenshot 2025-07-02 163951](https://github.com/user-attachments/assets/5c64f38a-7741-452f-b1b6-670957f659b0)

![Screenshot 2025-07-02 164039](https://github.com/user-attachments/assets/c07793c3-308d-41dd-9dd0-7dfa2a84ba36)

![Screenshot 2025-07-02 161106](https://github.com/user-attachments/assets/ef672def-a1c3-4c2f-a707-acd52f10ee8c)

![Screenshot 2025-07-02 161140](https://github.com/user-attachments/assets/7a2520f0-b5ad-4aef-8c04-144237369b8d)

![Screenshot 2025-07-02 165642](https://github.com/user-attachments/assets/7a794eaa-9a5a-4a11-8d0f-646bd387d875)


**Comment**:
ODIN improved OOD detection performance, achieving the highest AUC (0.86) among all tested methods.

