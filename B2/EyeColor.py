import csv
import os
import pathlib
import PIL
import torch
from PIL import Image
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import livelossplot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent_dir_path = pathlib.Path(__file__).parent.parent.resolve()
dataset_dir_path = os.path.join(parent_dir_path, "Datasets")


class CustomDataset(Dataset):
    def __init__(self, dataset_path, label_filepath, image_num, column_num):
        self.images = []
        self.labels = []
        convert_tensor = transforms.ToTensor()
        with open(label_filepath, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            input_csv_data = list(reader)
        for c in range(len(input_csv_data) - 1):
            label = int(input_csv_data[c + 1][column_num])
            image_filename = input_csv_data[c + 1][3]
            image_filepath = os.path.join(dataset_path, image_filename)
            if os.path.exists(image_filepath):
                image = Image.open(image_filepath)
                image = image.resize((64, 64), Image.ANTIALIAS)
                image = convert_tensor(image).to(device)
                self.images.append(image)
                self.labels.append(label)
            if len(self.images) >= image_num:
                break
        self.labels = torch.tensor(self.labels).to(device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class EyeColorDetector(Module):
    def __init__(self, shape, output_channel_size):
        super(EyeColorDetector, self).__init__()
        input_channel_size, m, n = shape
        self.detector = nn.Sequential(
            # input_channel_size * 64 * 64
            nn.Conv2d(input_channel_size, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 16 * 32 * 32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 32 * 16 * 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 64 * 8 * 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 128 * 4 * 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            # 256 * 2 * 2
        )
        self.linear = nn.Sequential(
            nn.Linear(256 * 2 * 2, 64 * 2 * 2),
            nn.Linear(64 * 2 * 2, output_channel_size)
        )

    def forward(self, x):
        x = self.detector(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def loss_cal(predict_prob, target_prob):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(predict_prob, target_prob)


train_set_path = os.path.join(dataset_dir_path, "cartoon_set", "img")
test_set_path = os.path.join(dataset_dir_path, "cartoon_set_test", "img")
train_label_path = os.path.join(dataset_dir_path, "cartoon_set", "labels.csv")
test_label_path = os.path.join(dataset_dir_path, "cartoon_set_test", "labels.csv")
training_image_num = 10000  # Number of images to load in training dataset
test_image_num = 2500  # Number of images to load in test dataset
label_column_num = 1  # Column number of the label in the label file
cartoon_train_set = CustomDataset(train_set_path, train_label_path, training_image_num, label_column_num)
cartoon_test_set = CustomDataset(test_set_path, test_label_path, test_image_num, label_column_num)
cartoon_train_loader = DataLoader(cartoon_train_set, num_workers=0, batch_size=128, shuffle=True)

epoch = 100
num_classes = 5
eye_color_detector = EyeColorDetector(cartoon_train_set[0][0].shape, num_classes).to(device)
optimizer = torch.optim.Adam(eye_color_detector.parameters(), lr=1e-3, weight_decay=1e-5)
loss_plot = livelossplot.PlotLosses()
for i in range(epoch):
    training_loss = 0
    training_accuracy = 0
    for data in cartoon_train_loader:
        images, target = data
        predict = eye_color_detector(images)
        optimizer.zero_grad()
        loss = loss_cal(predict, target)
        predict_label = torch.argmax(predict, dim=1)
        training_accuracy += torch.sum(predict_label == target).item() / len(target)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_accuracy /= len(cartoon_train_loader)
    correct = 0
    for data in cartoon_test_set:
        test_image, test_target = data
        test_predict = eye_color_detector(test_image[None, :, :, :])
        test_predict_label = torch.argmax(test_predict)
        if test_target == test_predict_label:
            correct += 1
    training_loss = training_loss / len(cartoon_train_loader)
    test_accuracy = correct / len(cartoon_test_set)
    loss_plot.update({"training_loss": training_loss, "training_accuracy": training_accuracy,
                      "test_accuracy": test_accuracy})
    print("Epoch: %s, Loss: %s" % (i, training_loss))
    print("Training_accuracy: %s" % training_accuracy)
    print("Test_accuracy: %s" % test_accuracy)
    print("---------------------")

loss_plot.send()
