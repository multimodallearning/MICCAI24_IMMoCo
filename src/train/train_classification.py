import sys

sys.path.appen("src/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from models.classification import get_model
from utils.classification_utils import multi_acc

# load train, val and test datasets
train_dataset = torch.load("Dataset/ClassificationData/dataset_train.pth")
val_dataset = torch.load("Dataset/ClassificationData/dataset_val.pth")

print("Train dataset:", train_dataset["images"].shape, train_dataset["labels"].shape)
train_img = train_dataset["images"][:2000]
train_label = train_dataset["labels"][:2000]
weights = train_dataset["label_weight"]

print("Val dataset:", val_dataset["images"].shape, val_dataset["labels"].shape)
val_img = val_dataset["images"][:300]
val_label = val_dataset["labels"][:300]


num_classes = train_label.unique().shape[0]
print("Number of classes:", num_classes)
model = get_model(num_classes)

model = model.cuda()

# freeze all layers but the fully connected layers
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# normalize images between 0 and 1 for each image alone
min_train = (
    train_img.view(train_img.shape[0], -1)
    .min(1)
    .values.unsqueeze(1)
    .unsqueeze(2)
    .unsqueeze(3)
)
max_train = (
    train_img.view(train_img.shape[0], -1)
    .max(1)
    .values.unsqueeze(1)
    .unsqueeze(2)
    .unsqueeze(3)
)
train_img = (train_img - min_train) / (max_train - min_train)

min_val = (
    val_img.view(val_img.shape[0], -1)
    .min(1)
    .values.unsqueeze(1)
    .unsqueeze(2)
    .unsqueeze(3)
)
max_val = (
    val_img.view(val_img.shape[0], -1)
    .max(1)
    .values.unsqueeze(1)
    .unsqueeze(2)
    .unsqueeze(3)
)
val_img = (val_img - min_val) / (max_val - min_val)

# one hot encode labels
train_label = F.one_hot(train_label, num_classes=train_label.unique().shape[0]).float()
val_label = F.one_hot(val_label, num_classes=val_label.unique().shape[0]).float()


# ResNet Specific normalization: https://pytorch.org/hub/pytorch_vision_resnet/
process = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
train_img = process(train_img)
val_img = process(val_img)


if torch.cuda.is_available():
    weights = weights.cuda()
    model = model.cuda()


# Hyperparameters
batch_size = 20
learning_rate = 0.001
num_epochs = 100

criterion = nn.CrossEntropyLoss(weight=weights.cuda())

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epochs):
    model.train()
    rand_idx = torch.randperm(train_img.shape[0]).view(-1, batch_size)

    for i in range(0, rand_idx.shape[0]):
        # Get the current batch
        batch_img = train_img[rand_idx[i]].cuda()
        batch_label = train_label[rand_idx[i]].cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_img)

        # Compute the loss
        loss = criterion(outputs, batch_label)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    rand_idx = torch.randperm(val_img.shape[0]).view(-1, batch_size)
    val_loss = 0
    val_acc = 0
    for i in range(0, rand_idx.shape[0]):
        with torch.no_grad():
            # Get the current batch
            batch_img = val_img[rand_idx[i]].cuda()
            batch_label = val_label[rand_idx[i]].cuda()

            # Forward pass
            outputs = model(batch_img)

            # Compute the loss
            loss = criterion(outputs, batch_label)
            acc = multi_acc(outputs, batch_label)

            val_loss += loss.item()
            val_acc += acc.item()

    # Print the loss every 100 batches
    val_loss /= rand_idx.shape[0]
    val_acc /= rand_idx.shape[0]

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%"
        )
# save the model
torch.save(model.state_dict(), "classification_model.pth")
