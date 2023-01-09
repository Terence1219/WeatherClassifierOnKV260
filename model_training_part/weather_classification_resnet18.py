import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(self.resnet18.fc.in_features, 3)
        self.resnet18.fc = self.fc

    def forward(self, x):
        x = self.resnet18(x)
        return x


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    # 定義數據目錄
    data_dir = 'C:\dataset'

    # 載入數據集
    dataset = ImageFolder(data_dir,transform=transform)
    
    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
   
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
 
    classes = ['cloudy','rainy','sunny']


    net = Net().to(device)
    class_weights = np.array([0.858,28.6175,0.5557])
    class_weights = torch.from_numpy(class_weights).float()
    criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
    optimizer = optim.Adam(net.parameters(), lr=0.001) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0
    save_path = 'C:\model\weather_resnet18.pth'
    epochs = 15
    
    train_acc = []
    val_acc = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_correct = 0
        train_total = 0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        acc = round(100 *train_correct / train_total,2)
        print('Epoch {number} training accuracy:{acc}%'.format(number=epoch+1, acc=acc))
        train_acc.append(acc)

        test_correct = 0
        test_total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            acc=round(100 *test_correct / test_total,2)
            print('Validation accuracy:{acc}%'.format(acc=acc))
            val_acc.append(acc)
            if acc > best_acc:
                tqdm.write('Model saved.')
                torch.save(net.state_dict(), save_path)
                best_acc = acc

    x = np.arange(1,epochs+1)
    plt.figure(1)
    plt.plot(x,train_acc,label='train')
    plt.plot(x,val_acc,label='val')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    ax=plt.gca()
    ax.set_title('accuracy')
    plt.savefig('/model/resnet18_acc.png') 