import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from tqdm.auto import tqdm


class DataPreparation:
    def __init__(self, batch_size=512):
        self.batch_size = batch_size
        self.set_seeds()
        self.prepare_transforms()
        self.load_datasets()
        self.create_dataloaders()

    def set_seeds(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def prepare_transforms(self):
        self.transforms = {
            "train": Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
            ]),
            "test": ToTensor(),
        }

    def load_datasets(self):
        self.dataset = {}
        for split in ["train", "test"]:
            self.dataset[split] = CIFAR10(
                root="data/cifar10",
                train=(split == "train"),
                download=True,
                transform=self.transforms[split],
            )

    def create_dataloaders(self):
        self.dataflow = {}
        for split in ['train', 'test']:
            self.dataflow[split] = DataLoader(
                self.dataset[split],
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=0,
                pin_memory=True,
            )


class VGG(nn.Module):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self):
        super().__init__()
        self.backbone = self.create_backbone()
        self.classifier = nn.Linear(512, 10)

    def create_backbone(self):
        layers = []
        counts = defaultdict(int)
        in_channels = 3

        def add(name, layer):
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        for x in self.ARCH:
            if x != 'M':
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                add("pool", nn.MaxPool2d(2))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class Trainer:
    def __init__(self, model, dataflow, num_epochs=2):
        self.model = model
        self.dataflow = dataflow
        self.num_epochs = num_epochs
        self.setup_optimization()

    def setup_optimization(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=0.4,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = len(self.dataflow["train"])
        lr_lambda = lambda step: np.interp(
            [step / steps_per_epoch],
            [0, self.num_epochs * 0.3, self.num_epochs],
            [0, 1, 0]
        )[0]
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):
        self.model.train()
        for inputs, targets in tqdm(self.dataflow["train"], desc='train', leave=False):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        num_samples = num_correct = 0
        with torch.inference_mode():
            for inputs, targets in tqdm(self.dataflow["test"], desc="eval", leave=False):
                outputs = self.model(inputs).argmax(dim=1)
                num_samples += targets.size(0)
                num_correct += (outputs == targets).sum()
        return (num_correct / num_samples * 100).item()

    def train(self):
        for epoch_num in tqdm(range(1, self.num_epochs + 1)):
            self.train_epoch()
            metric = self.evaluate()
            print(f"epoch {epoch_num}:", metric)


class Visualizer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def visualize(self, num_samples=40):
        plt.figure(figsize=(20, 10))
        for index in range(num_samples):
            image, label = self.dataset["test"][index]

            self.model.eval()
            with torch.inference_mode():
                pred = self.model(image.unsqueeze(dim=0))
                pred = pred.argmax(dim=1)

            image = image.permute(1, 2, 0)
            pred_class_name = self.dataset["test"].classes[pred]
            label_class_name = self.dataset["test"].classes[label]

            plt.subplot(4, 10, index + 1)
            plt.imshow(image)
            plt.title(f"pred: {pred_class_name}\nlabel: {label_class_name}")
            plt.axis("off")
        plt.show()


# Main execution
if __name__ == "__main__":
    data_prep = DataPreparation()
    model = VGG()
    trainer = Trainer(model, data_prep.dataflow)
    trainer.train()
    visualizer = Visualizer(model, data_prep.dataset)
    visualizer.visualize()