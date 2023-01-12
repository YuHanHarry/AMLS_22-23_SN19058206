1. description of the organization of your project:
Before run the code upload 
•cartoon_set
• celeba
• cartoon_set_test
• celeba_test
inside Datasets 

A1 A2 B1 B2 include individual tasks.

2.the role of each file:
A1: gender detection. A2: Emotion detection. B1:Face shape recognition B2: Eye color recognition.
the lab3_data file could abstract the processed image data for task A.

3.the packages required to run your code:
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
import numpy as np
from keras_preprocessing import image
import cv2
import dlib
import tensorflow._api.v2.compat.v1 as tf
