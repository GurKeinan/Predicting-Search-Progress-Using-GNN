import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn import preprocessing
from torch.autograd import Variable
import torchviz
import matplotlib.pyplot as plt
from datetime import datetime


class DataSet:
    """
    a general DataSet
    """
    def __init__(self,
                 features_path_data,
                 labels_path_data,
                 features_columns,
                 labels_columns
                ):

        # load csv to pd.DataFrame
        self.df = pd.read_csv(features_path_data, skip_blank_lines=True, header=0, usecols=features_columns)
        
        self.features_columns = features_columns
        self.labels_columns = labels_columns
        max_path = pd.read_csv(features_path_data, skip_blank_lines=True, header=0, usecols=["node_max"])
        node_num = pd.read_csv(features_path_data, skip_blank_lines=True, header=0, usecols=["level_1#"])
        self.y=[]

        for i in range(len(node_num)):
            self.y.append(node_num["level_1#"][i]/max_path["node_max"][i]*100)
        self.len = len(self.y)

    def Dataset(self, K, num_of_features):
        temp_padding = []
        for i in range(num_of_features):
            temp_padding.append(0)
        temp_list_features = []
        temp_list_labels = []
        main_list= []
        for i in range(K):
            temp_list_features.append(temp_padding)
        template_features = temp_list_features.copy()
        for i in range(self.len):
            new_instance_list_features = list(self.df[self.features_columns][i:(i+1)].values[0])
            if new_instance_list_features[0] == 0:
                temp_list_features = template_features.copy()
            temp_list_features.pop(0)
            temp_list_features.append(new_instance_list_features)
            temp_list_labels.append(self.y[i])
            main_list.append((temp_list_features.copy(), temp_list_labels.copy()))
            temp_list_labels.pop(0)
        return main_list   
    
    def final_dataset(self, K, num_of_features):
        main_list = self.Dataset(K, num_of_features)
        n_instances = len(main_list)
        x = torch.zeros((n_instances, K, num_of_features))
        y = torch.zeros(n_instances)
        for i in range(n_instances):
            x[i] = torch.tensor(main_list[i][0])
            y[i] = torch.tensor(main_list[i][1])
        
        return (x,y)
    
    def __len__(self):
        return len(self.df[:])
    
def tensor_to_list(x, y):
    dataset = []
    for i in range(len(y)):
        dataset.append((x[i],y[i]))
    return(dataset)

    
if __name__ == "__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """features_columns = [
        "level_1#", "level_1_F", "level_1_H", "level_1_G", "level_1_BF",
        "level_2#", "level_2_F", "level_2_H", "level_2_G", "level_2_BF",
        "level_3#", "level_3_F", "level_3_H", "level_3_G", "level_3_BF",
        "level_4#", "level_4_F", "level_4_H", "level_4_G", "level_4_BF",
        "level_5#", "level_5_F", "level_5_H", "level_5_G", "level_5_BF",
        "level_6#", "level_6_F", "level_6_H", "level_6_G", "level_6_BF",
        "level_7#", "level_7_F", "level_7_H", "level_7_G", "level_7_BF",
        "level_8#", "level_8_F", "level_8_H", "level_8_G", "level_8_BF",
        "level_9#", "level_9_F", "level_9_H", "level_9_G", "level_9_BF",
        "level_10#", "level_10_F", "level_10_H", "level_10_G", "level_10_BF",
        "H0", "H_min", "last_H_min_update",	"f_max"
    ]"""

    features_columns = [
        "level_1#", "level_1_F", "level_1_H", "level_1_G", "level_1_BF",
        "level_2#", "level_2_F", "level_2_H", "level_2_G", "level_2_BF",
        "level_3#", "level_3_F", "level_3_H", "level_3_G", "level_3_BF",
        "H0", "H_min", "last_H_min_update",	"f_max"
    ]


    labels_columns = ["y"]
    expeirment_type = "aster_with_hff"
    instances = {}

    instances["miconic_01"] = "data/preprocessing/"+expeirment_type+"/miconic/task01.pddl.csv"
    instances["miconic_02"] = "data/preprocessing/"+expeirment_type+"/miconic/task02.pddl.csv"
    instances["miconic_03"] = "data/preprocessing/"+expeirment_type+"/miconic/task03.pddl.csv"
    instances["miconic_04"] = "data/preprocessing/"+expeirment_type+"/miconic/task04.pddl.csv"
    instances["miconic_05"] = "data/preprocessing/"+expeirment_type+"/miconic/task05.pddl.csv"
    instances["miconic_06"] = "data/preprocessing/"+expeirment_type+"/miconic/task06.pddl.csv"
    instances["miconic_07"] = "data/preprocessing/"+expeirment_type+"/miconic/task07.pddl.csv"
    instances["miconic_08"] = "data/preprocessing/"+expeirment_type+"/miconic/task08.pddl.csv"
    instances["miconic_09"] = "data/preprocessing/"+expeirment_type+"/miconic/task09.pddl.csv"
    instances["miconic_10"] = "data/preprocessing/"+expeirment_type+"/miconic/task10.pddl.csv"
    instances["miconic_11"] = "data/preprocessing/"+expeirment_type+"/miconic/task11.pddl.csv"
    instances["miconic_12"] = "data/preprocessing/"+expeirment_type+"/miconic/task12.pddl.csv"
    instances["miconic_13"] = "data/preprocessing/"+expeirment_type+"/miconic/task13.pddl.csv"
    instances["miconic_14"] = "data/preprocessing/"+expeirment_type+"/miconic/task14.pddl.csv"
    instances["miconic_15"] = "data/preprocessing/"+expeirment_type+"/miconic/task15.pddl.csv"
    instances["miconic_16"] = "data/preprocessing/"+expeirment_type+"/miconic/task16.pddl.csv"
    instances["miconic_17"] = "data/preprocessing/"+expeirment_type+"/miconic/task17.pddl.csv"
    instances["miconic_18"] = "data/preprocessing/"+expeirment_type+"/miconic/task18.pddl.csv"
    instances["miconic_19"] = "data/preprocessing/"+expeirment_type+"/miconic/task19.pddl.csv"
    instances["miconic_20"] = "data/preprocessing/"+expeirment_type+"/miconic/task20.pddl.csv"
    instances["miconic_21"] = "data/preprocessing/"+expeirment_type+"/miconic/task21.pddl.csv"
    instances["miconic_22"] = "data/preprocessing/"+expeirment_type+"/miconic/task22.pddl.csv"
    instances["miconic_23"] = "data/preprocessing/"+expeirment_type+"/miconic/task23.pddl.csv"
    instances["miconic_24"] = "data/preprocessing/"+expeirment_type+"/miconic/task24.pddl.csv"
    instances["miconic_25"] = "data/preprocessing/"+expeirment_type+"/miconic/task25.pddl.csv"
    instances["miconic_26"] = "data/preprocessing/"+expeirment_type+"/miconic/task26.pddl.csv"
    instances["miconic_27"] = "data/preprocessing/"+expeirment_type+"/miconic/task27.pddl.csv"
    instances["miconic_28"] = "data/preprocessing/"+expeirment_type+"/miconic/task28.pddl.csv"
    instances["miconic_29"] = "data/preprocessing/"+expeirment_type+"/miconic/task29.pddl.csv"
    instances["miconic_30"] = "data/preprocessing/"+expeirment_type+"/miconic/task30.pddl.csv"

    original_path = os.getcwd()
    K=30
    num_of_featuers = len(features_columns)
    for instance in instances:
        #checking if we already did the file or we are missing the output
        if os.path.isfile("/temp/"+instance+"_x_.pt") or os.path.isfile(instances[instance]) == False:
            continue
        path = instances[instance]
        features_path_data_train = path
        labels_path_data_train = path
        train = DataSet(features_path_data_train, labels_path_data_train, features_columns, labels_columns)
        if (train.len < 1000 or train.len>1000000):
            continue
        x_train, y_train = train.final_dataset(K, num_of_featuers)
        torch.save(x_train, "temp/"+instance+"_x_.pt")
        torch.save(y_train, "temp/"+instance+"_y_.pt")
        print(instance)

