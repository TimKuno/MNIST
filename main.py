import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# load and preprocess the data
kwargs = {}  # empty since no cuda id used
train_data = torch.utils.data.DataLoader(
    datasets.MNIST('train', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST('test', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)


# defining the model
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 2)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.conv_dropout(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(out)

        out = out.view(-1, 320)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


model = Cnn()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)


# defining the training of the model
def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} [{100 * batch_id / len(train_data):.2f}%]'
              f'\t Loss: {loss.item():.6f}')


# defining the testing of the modell
def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        data = Variable(data)
        target = Variable(target)
        out = model(data)
        loss += F.nll_loss(out, target, reduction='none').data[0]
        prediction = out.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).sum()
    loss = loss / len(test_data.dataset)
    print(f'Accuracy: {100 * correct / len(test_data.dataset):.2f}% \t Average loss: {loss:0.6f}')


# train the model in order to test it afterwards for the evaluation
for epoch in range(1, 20):
    train(epoch)
test()  # it would also be possible to evaluate the model after each epoch
