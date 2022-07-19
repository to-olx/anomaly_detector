import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import pandas as pd
from PIL import Image
import cv2
import numpy as np


transform = transforms.Compose([transforms.Resize(255),
                                 transforms.ToTensor()])


dataset = datasets.ImageFolder('data/isolated_images/', transform=transform)
print(dataset)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
print(dataloader)
# ======================================================= Images pillow ========================================================== #

def normalize(arr):
    """
    Linear normalization
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr


def pipe():
    print('> Creating Isolated Images')
    data = pd.read_csv("data/annotations/instances_classes.csv")
    isolated_file_count = 0
    file_names = []
    for i, row in data.iterrows():
        # Open the image
        image = Image.open(f"data/task_road_sign_classes_backup_2022_06_23_09_49_27/data/{row['file_name']}")
        # Crop the image
        file_name = f'{isolated_file_count}'+'.'+row.file_name.split('.')[1]
        file_names.append(file_name)
        cropped = image.crop((row['x1'], row['y1'], row['x2'], row['y2']))
        rows=100
        cols=100
        resized = cropped.resize((rows, cols))
        arr = np.array(resized)
        new_img = Image.fromarray(normalize(arr).astype('uint8'))
        new_img.save(f"data/isolated_images/{file_name}")
        isolated_file_count+=1
    print('.. Done Creating Isolated Images')
    classes_df = pd.read_csv('data/annotations/instances_classes.csv')
    labeled_df = pd.read_csv('data/annotations/instances_labeled.csv')
    classes_df['isolated_file_name'] = file_names
    classes_df.label = classes_df.label.map(
        {
            1: "guidance",
            4: "warning",
            2: "regulation" 
        }
    )
    return classes_df

pipe()



# ======================================================= CREATE DATASETS ========================================================= #

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# ====================================================== SET BATCH SIZE ============================================================ #

batch_size = 64

# ====================================================== ADD DATA INTO DATALOADER ================================================== #

# Create data loaders.
train_dataloader = DataLoader(dataset, batch_size=batch_size)
test_dataloader = DataLoader(dataset, batch_size=batch_size)

for X, y in dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# ====================================================== CHECK IF DEVICE HAS GPU ================================================== #

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ====================================================== CREATE PREDICTIVE MODEL ================================================== #

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# ====================================================== CREATE LOSS FUNCTION AND OPTIMIZER ======================================= #

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# ====================================================== TRAIN & TEST MODEL ======================================================= #


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss {test_loss:>8f}\n")


epochs = 30

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")


# =================================================================== SAVE TRAINED MODEL ==================================================================== #

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# =================================================================== LOAD TRAINED MODEL ==================================================================== #


model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# =================================================================== MAKE PREDICTION MODEL ================================================================= #


classes = {
    "guidance",
    "warning",
    "regulation"
}


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    print(x, y)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
