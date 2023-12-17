import cv2
from matplotlib import pyplot as plt
import torchvision
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn
from PIL import Image


class mnist_inference:
    def __init__(self):
        self.eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.model = torchvision.models.vgg19_bn(num_classes=10)
        self.model.load_state_dict(
            torch.load("mnist_net.pth", map_location=torch.device("cpu"))
        )
        self.model.eval()
        self.classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    def add_image(self, image):
        self.image = cv2.imread(image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.show_image = cv2.resize(self.image, (128, 128))

    def inference(self):
        input = self.eval_transforms(cv2.resize(self.image, (32, 32)))
        result = self.model(input.unsqueeze(0))
        result = torch.nn.functional.softmax(result, dim=1)
        return result

    def accuracy_loss(self):
        result = Image.open("fig1.png")
        plt.imshow(result)
        plt.show()


class cat_dog_inference:
    def __init__(self):
        self.eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.model = torchvision.models.resnet50(pretrained=True)
        nr_filters = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(nr_filters, 1), nn.Sigmoid())
        self.model.load_state_dict(
            torch.load("cat_dog_net.pth", map_location=torch.device("cpu"))
        )
        self.model.eval()

    def show_image(self):
        cat = Image.open("Dataset_OpenCvDl_Hw2/Q5/validation_dataset/Cat/999.jpg")
        dog = Image.open("Dataset_OpenCvDl_Hw2/Q5/validation_dataset/Dog/374.jpg")
        plt.subplot(1, 2, 1)
        plt.imshow(cat)
        plt.title("Cat")
        plt.subplot(1, 2, 2)
        plt.imshow(dog)
        plt.title("Dog")
        plt.show()

    def add_image(self, image):
        self.image = cv2.imread(image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.show_image = cv2.resize(self.image, (224, 224))

    def inference(self):
        input = self.eval_transforms(self.show_image)
        result = self.model(input.unsqueeze(0))
        result = result.detach().numpy()[0][0]
        return result

    def compare(self):
        x = ["random erase", "without random erase"]
        y = [0.956, 0.97]
        plt.bar(x, y)
        plt.title("Comparison")
        plt.xlabel("Method")
        plt.ylabel("Accuracy")
        plt.show()
