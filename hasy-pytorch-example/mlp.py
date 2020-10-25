# core modules
import os

# 3rd party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

# internal modules
import hasy_tools


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Kevin Bryson
        """
        nn.Conv2d(input channel, convolutional features, square kernel size, stride)
        input channel = image
        convolutional features = number of pixels (each is 32x32)
        square kernel size = RGB in image represented as matrix (more features to extract)
        """
        
        
        # self.fc1 = nn.Linear(32*32, 100)
        # self.fc2 = nn.Linear(100, 369)
        ## Ng model
        self.conv1 = nn.Conv2d(in_channels =1, out_channels=32, kernel_size=3)
        self.pool1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels =32, out_channels=64, kernel_size=3)
        self.pool2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features = 2*2*128, out_features=760)
    
        self.fc2 = nn.Linear(in_features=760, out_features= 369)
        self.dropout = nn.Dropout2d(.25)
 
        
        
    def forward(self, x):
        #  Ng Model
        x = self.pool1(self.conv1(x))
        x = F.relu(F.max_pool2d(x,2))
        x = self.pool2(self.conv2(x))
        x = F.relu(F.max_pool2d(x,2))
        x = self.pool3(self.conv3(x))
        x = F.relu(F.max_pool2d(x,2))   

        # Fully connected with dropout  from Ng model
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) 


        return F.log_softmax(x, dim=1)

class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Creating this Net to avoid overalap with the main Net for better merging.
          in_channels_layer1 = 1
        out_channels_layer1 = 32
        in_channels_layer2 = out_channels_layer1
        out_channels_layer2 = 64
        in_channels_layer3 = out_channels_layer2
        out_channels_layer3 = 256
        kernel_size = 3
        out_features_layer1 = 712
        out_features = 369
        ## Ng model
        self.conv1 = nn.Conv2d(in_channels =in_channels_layer1, out_channels=out_channels_layer1, kernel_size=kernel_size)
        self.pool1 = nn.BatchNorm2d(out_channels_layer1)
        self.conv2 = nn.Conv2d(in_channels =in_channels_layer2, out_channels=out_channels_layer2, kernel_size=kernel_size)
        self.pool2 = nn.BatchNorm2d(out_channels_layer2)
        self.conv3 = nn.Conv2d(in_channels=in_channels_layer3, out_channels=out_channels_layer3, kernel_size=kernel_size)
        self.pool3 = nn.BatchNorm2d(out_channels_layer3)

        self.fc1 = nn.Linear(in_features = 2*2*out_channels_layer3, out_features=out_features_layer1)
        self.fcbn = nn.BatchNorm1d(out_features_layer1)
        self.fc2 = nn.Linear(in_features=out_features_layer1, out_features= out_features)
        # Drop out rate should be controllable as well
        self.dropout = nn.Dropout2d(.25)
 
        
        
   def forward(self, x):
        #  Ng Model
        x = self.pool1(self.conv1(x))
        x = F.relu(F.max_pool2d(x,2))
        x = self.pool2(self.conv2(x))
        x = F.relu(F.max_pool2d(x,2))
        x = self.pool3(self.conv3(x))
        x = F.relu(F.max_pool2d(x,2))   

        # Fully connected with dropout  from Ng model
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fcbn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) 


        return F.log_softmax(x, dim=1)

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


class HASY(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.data = hasy_tools.load_data(mode='fold-2') # see function comments
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data['x_train'][index], self.data['y_train'][index]
        else:
            img, target = self.data['x_test'][index], self.data['y_test'][index]

        target = target[0]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        # img.squeeze()
        return img, target

    def __len__(self):
        if self.train:
            return len(self.data['y_train'])
        else:
            return len(self.data['y_test'])


def main():
    PATH = "hasy-visionTeX-model.pt"
    # Training settings
    batch_size = 50
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 1
    log_interval = 10
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print ("Device",device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        HASY('../data', train=True, download=True,
             transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    hasy_tools.load_data()

    test_loader = torch.utils.data.DataLoader(
        HASY('../data', train=False,
             transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    
    torch.save(model.state_dict(), PATH)
    '''
    when loading this make sure to:
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model    
    '''

if __name__ == '__main__':
    main()
