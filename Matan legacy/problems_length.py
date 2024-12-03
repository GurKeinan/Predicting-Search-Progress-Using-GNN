import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import utils
import src.pytorch_utils as ptu
import sklearn
from sklearn import preprocessing
from torch.autograd import Variable
import torchviz
import matplotlib.pyplot as plt
from datetime import datetime

def select_k_nodes(dataset, labels,  k):
    rand = torch.rand(k)
    rand = rand * dataset.shape[0]
    for i in range(len(rand)):
        if (i==0):
            new_dataset = dataset[int(rand[i])]
            new_labels = labels[int(rand[i])]
            new_dataset = new_dataset.unsqueeze(0)
            new_labels = new_labels.unsqueeze(0)
        else:
            temp_labels = labels[int(rand[i])]
            temp_dataset = dataset[int(rand[i])]
            temp_dataset = temp_dataset.unsqueeze(0)
            temp_labels = temp_labels.unsqueeze(0)
            new_dataset = torch.cat((new_dataset, temp_dataset), 0)
            new_labels = torch.cat((new_labels, temp_labels), 0)
    return new_dataset, new_labels



instances = {}
path = "data/ready_data/astar_hff"
algorithm = path.split("/")
algorithm = algorithm[-1]
main_path = "/data/preprocessing/astar_hff"
domains =[
    ("airport", 30),
    ("blocks", 35),
    ("depot", 22),
    ("elevators", 30),
    ("freecell", 20),
    ("gripper", 20),
    ("logistics", 28),
    ("miconic", 30),
    ("movie", 30),
    ("openstacks", 30),
    ("parcprinter", 30),
    ("pegsol", 30),
    ("psr-small", 50),
    ("rovers", 30),
    ("satellite", 20),
    ("scanalyzer", 30),
    ("sokoban", 30),
    ("tpp", 30),
    ("transport", 30),
    ("woodworking", 30),
    ("zenotravel", 20),
]

paths = []
for domain in domains:
    temp_domain = domain[0]
    num_instances = domain[1]
    for instance in range(num_instances):
        if (instance<9):
            str_instance = "0"+str(instance+1)
        else:
            str_instance = str(instance+1)
        paths.append((temp_domain+"_"+str_instance,\
                path+"/"+temp_domain+"_"+str_instance+"_x_.pt"))

full_dataset = True
k = 1000
train_x = None
train_y = None
test_x = None
test_y = None
length = []
for path_tuple in paths:
    instance = path_tuple[0]
    if os.path.isfile(os.getcwd()+"/"+path+"/"+instance+"_y_.pt") == False:
        continue
    temp_y = torch.load(path+"/"+instance+"_y_.pt")
    length.append((instance, temp_y.shape[0]))
x_df = pd.DataFrame(length)
x_df.to_csv("problem_length_"+algorithm+".csv")

        