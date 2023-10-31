import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

net = torchvision.models.vgg19_bn(num_classes=len(classes))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 10
    
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    print("Start Training")

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    for epoch in range(40):
        print(f"Epoch: {epoch + 1}")
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
        train_loss.append(running_loss / len(trainloader))
        test_loss.append(val_loss / len(testloader))
        train_acc.append(train_correct / train_total)
        test_acc.append(val_correct / val_total)
        
        print(f"[{epoch + 1}, {i + 1:5d}]")
        print(f"\tloss: {train_loss[-1]:.3f} val loss: {test_loss[-1]:.3f}")
        print(f"\taccuracy: {train_acc[-1]:.3f} val accuracy: {test_acc[-1]:.3f}")

    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(train_loss, label="Train")
    axs[0].plot(test_loss, label="Test")
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc="upper right")
    axs[0].set_title("Loss")
    axs[0].grid(True)

    axs[1].plot(train_acc, label="Train")
    axs[1].plot(test_acc, label="Test")
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc="upper left")
    axs[1].set_title("Accuracy")
    axs[1].grid(True)

    print(train_loss)
    print(test_loss)
    print(train_acc)
    print(test_acc)

    print("Finished Training")
    torch.save(net.state_dict(), "./cifar_net.pth")
