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
    expeirment_type = "gbfs_lmcut"
    instances = {}

    


    instances["tpp_01"] = "data/preprocessing/"+expeirment_type+"/tpp/task01.pddl.csv"
    instances["tpp_02"] = "data/preprocessing/"+expeirment_type+"/tpp/task02.pddl.csv"
    instances["tpp_03"] = "data/preprocessing/"+expeirment_type+"/tpp/task03.pddl.csv"
    instances["tpp_04"] = "data/preprocessing/"+expeirment_type+"/tpp/task04.pddl.csv"
    instances["tpp_05"] = "data/preprocessing/"+expeirment_type+"/tpp/task05.pddl.csv"
    instances["tpp_06"] = "data/preprocessing/"+expeirment_type+"/tpp/task06.pddl.csv"
    instances["tpp_07"] = "data/preprocessing/"+expeirment_type+"/tpp/task07.pddl.csv"
    instances["tpp_08"] = "data/preprocessing/"+expeirment_type+"/tpp/task08.pddl.csv"
    instances["tpp_09"] = "data/preprocessing/"+expeirment_type+"/tpp/task09.pddl.csv"
    instances["tpp_10"] = "data/preprocessing/"+expeirment_type+"/tpp/task10.pddl.csv"
    instances["tpp_11"] = "data/preprocessing/"+expeirment_type+"/tpp/task11.pddl.csv"
    instances["tpp_12"] = "data/preprocessing/"+expeirment_type+"/tpp/task12.pddl.csv"
    instances["tpp_13"] = "data/preprocessing/"+expeirment_type+"/tpp/task13.pddl.csv"
    instances["tpp_14"] = "data/preprocessing/"+expeirment_type+"/tpp/task14.pddl.csv"
    instances["tpp_15"] = "data/preprocessing/"+expeirment_type+"/tpp/task15.pddl.csv"
    instances["tpp_16"] = "data/preprocessing/"+expeirment_type+"/tpp/task16.pddl.csv"
    instances["tpp_17"] = "data/preprocessing/"+expeirment_type+"/tpp/task17.pddl.csv"
    instances["tpp_18"] = "data/preprocessing/"+expeirment_type+"/tpp/task18.pddl.csv"
    instances["tpp_19"] = "data/preprocessing/"+expeirment_type+"/tpp/task19.pddl.csv"
    instances["tpp_20"] = "data/preprocessing/"+expeirment_type+"/tpp/task20.pddl.csv"
    instances["tpp_21"] = "data/preprocessing/"+expeirment_type+"/tpp/task21.pddl.csv"
    instances["tpp_22"] = "data/preprocessing/"+expeirment_type+"/tpp/task22.pddl.csv"
    instances["tpp_23"] = "data/preprocessing/"+expeirment_type+"/tpp/task23.pddl.csv"
    instances["tpp_24"] = "data/preprocessing/"+expeirment_type+"/tpp/task24.pddl.csv"
    instances["tpp_25"] = "data/preprocessing/"+expeirment_type+"/tpp/task25.pddl.csv"
    instances["tpp_26"] = "data/preprocessing/"+expeirment_type+"/tpp/task26.pddl.csv"
    instances["tpp_27"] = "data/preprocessing/"+expeirment_type+"/tpp/task27.pddl.csv"
    instances["tpp_28"] = "data/preprocessing/"+expeirment_type+"/tpp/task28.pddl.csv"
    instances["tpp_29"] = "data/preprocessing/"+expeirment_type+"/tpp/task29.pddl.csv"
    instances["tpp_30"] = "data/preprocessing/"+expeirment_type+"/tpp/task30.pddl.csv"

    instances["transport_01"] = "data/preprocessing/"+expeirment_type+"/transport/task01.pddl.csv"
    instances["transport_02"] = "data/preprocessing/"+expeirment_type+"/transport/task02.pddl.csv"
    instances["transport_03"] = "data/preprocessing/"+expeirment_type+"/transport/task03.pddl.csv"
    instances["transport_04"] = "data/preprocessing/"+expeirment_type+"/transport/task04.pddl.csv"
    instances["transport_05"] = "data/preprocessing/"+expeirment_type+"/transport/task05.pddl.csv"
    instances["transport_06"] = "data/preprocessing/"+expeirment_type+"/transport/task06.pddl.csv"
    instances["transport_07"] = "data/preprocessing/"+expeirment_type+"/transport/task07.pddl.csv"
    instances["transport_08"] = "data/preprocessing/"+expeirment_type+"/transport/task08.pddl.csv"
    instances["transport_09"] = "data/preprocessing/"+expeirment_type+"/transport/task09.pddl.csv"
    instances["transport_10"] = "data/preprocessing/"+expeirment_type+"/transport/task10.pddl.csv"
    instances["transport_11"] = "data/preprocessing/"+expeirment_type+"/transport/task11.pddl.csv"
    instances["transport_12"] = "data/preprocessing/"+expeirment_type+"/transport/task12.pddl.csv"
    instances["transport_13"] = "data/preprocessing/"+expeirment_type+"/transport/task13.pddl.csv"
    instances["transport_14"] = "data/preprocessing/"+expeirment_type+"/transport/task14.pddl.csv"
    instances["transport_15"] = "data/preprocessing/"+expeirment_type+"/transport/task15.pddl.csv"
    instances["transport_16"] = "data/preprocessing/"+expeirment_type+"/transport/task16.pddl.csv"
    instances["transport_17"] = "data/preprocessing/"+expeirment_type+"/transport/task17.pddl.csv"
    instances["transport_18"] = "data/preprocessing/"+expeirment_type+"/transport/task18.pddl.csv"
    instances["transport_19"] = "data/preprocessing/"+expeirment_type+"/transport/task19.pddl.csv"
    instances["transport_20"] = "data/preprocessing/"+expeirment_type+"/transport/task20.pddl.csv"
    instances["transport_21"] = "data/preprocessing/"+expeirment_type+"/transport/task21.pddl.csv"
    instances["transport_22"] = "data/preprocessing/"+expeirment_type+"/transport/task22.pddl.csv"
    instances["transport_23"] = "data/preprocessing/"+expeirment_type+"/transport/task23.pddl.csv"
    instances["transport_24"] = "data/preprocessing/"+expeirment_type+"/transport/task24.pddl.csv"
    instances["transport_25"] = "data/preprocessing/"+expeirment_type+"/transport/task25.pddl.csv"
    instances["transport_26"] = "data/preprocessing/"+expeirment_type+"/transport/task26.pddl.csv"
    instances["transport_27"] = "data/preprocessing/"+expeirment_type+"/transport/task27.pddl.csv"
    instances["transport_28"] = "data/preprocessing/"+expeirment_type+"/transport/task28.pddl.csv"
    instances["transport_29"] = "data/preprocessing/"+expeirment_type+"/transport/task29.pddl.csv"
    instances["transport_30"] = "data/preprocessing/"+expeirment_type+"/transport/task30.pddl.csv"


    instances["woodworking_01"] = "data/preprocessing/"+expeirment_type+"/woodworking/task01.pddl.csv"
    instances["woodworking_02"] = "data/preprocessing/"+expeirment_type+"/woodworking/task02.pddl.csv"
    instances["woodworking_03"] = "data/preprocessing/"+expeirment_type+"/woodworking/task03.pddl.csv"
    instances["woodworking_04"] = "data/preprocessing/"+expeirment_type+"/woodworking/task04.pddl.csv"
    instances["woodworking_05"] = "data/preprocessing/"+expeirment_type+"/woodworking/task05.pddl.csv"
    instances["woodworking_06"] = "data/preprocessing/"+expeirment_type+"/woodworking/task06.pddl.csv"
    instances["woodworking_07"] = "data/preprocessing/"+expeirment_type+"/woodworking/task07.pddl.csv"
    instances["woodworking_08"] = "data/preprocessing/"+expeirment_type+"/woodworking/task08.pddl.csv"
    instances["woodworking_09"] = "data/preprocessing/"+expeirment_type+"/woodworking/task09.pddl.csv"
    instances["woodworking_10"] = "data/preprocessing/"+expeirment_type+"/woodworking/task10.pddl.csv"
    instances["woodworking_11"] = "data/preprocessing/"+expeirment_type+"/woodworking/task11.pddl.csv"
    instances["woodworking_12"] = "data/preprocessing/"+expeirment_type+"/woodworking/task12.pddl.csv"
    instances["woodworking_13"] = "data/preprocessing/"+expeirment_type+"/woodworking/task13.pddl.csv"
    instances["woodworking_14"] = "data/preprocessing/"+expeirment_type+"/woodworking/task14.pddl.csv"
    instances["woodworking_15"] = "data/preprocessing/"+expeirment_type+"/woodworking/task15.pddl.csv"
    instances["woodworking_16"] = "data/preprocessing/"+expeirment_type+"/woodworking/task16.pddl.csv"
    instances["woodworking_17"] = "data/preprocessing/"+expeirment_type+"/woodworking/task17.pddl.csv"
    instances["woodworking_18"] = "data/preprocessing/"+expeirment_type+"/woodworking/task18.pddl.csv"
    instances["woodworking_19"] = "data/preprocessing/"+expeirment_type+"/woodworking/task19.pddl.csv"
    instances["woodworking_20"] = "data/preprocessing/"+expeirment_type+"/woodworking/task20.pddl.csv"
    instances["woodworking_21"] = "data/preprocessing/"+expeirment_type+"/woodworking/task21.pddl.csv"
    instances["woodworking_22"] = "data/preprocessing/"+expeirment_type+"/woodworking/task22.pddl.csv"
    instances["woodworking_23"] = "data/preprocessing/"+expeirment_type+"/woodworking/task23.pddl.csv"
    instances["woodworking_24"] = "data/preprocessing/"+expeirment_type+"/woodworking/task24.pddl.csv"
    instances["woodworking_25"] = "data/preprocessing/"+expeirment_type+"/woodworking/task25.pddl.csv"
    instances["woodworking_26"] = "data/preprocessing/"+expeirment_type+"/woodworking/task26.pddl.csv"
    instances["woodworking_27"] = "data/preprocessing/"+expeirment_type+"/woodworking/task27.pddl.csv"
    instances["woodworking_28"] = "data/preprocessing/"+expeirment_type+"/woodworking/task28.pddl.csv"
    instances["woodworking_29"] = "data/preprocessing/"+expeirment_type+"/woodworking/task29.pddl.csv"
    instances["woodworking_30"] = "data/preprocessing/"+expeirment_type+"/woodworking/task30.pddl.csv"


    instances["zenotravel_01"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task01.pddl.csv"
    instances["zenotravel_02"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task02.pddl.csv"
    instances["zenotravel_03"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task03.pddl.csv"
    instances["zenotravel_04"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task04.pddl.csv"
    instances["zenotravel_05"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task05.pddl.csv"
    instances["zenotravel_06"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task06.pddl.csv"
    instances["zenotravel_07"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task07.pddl.csv"
    instances["zenotravel_08"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task08.pddl.csv"
    instances["zenotravel_09"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task09.pddl.csv"
    instances["zenotravel_10"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task10.pddl.csv"
    instances["zenotravel_11"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task11.pddl.csv"
    instances["zenotravel_12"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task12.pddl.csv"
    instances["zenotravel_13"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task13.pddl.csv"
    instances["zenotravel_14"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task14.pddl.csv"
    instances["zenotravel_15"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task15.pddl.csv"
    instances["zenotravel_16"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task16.pddl.csv"
    instances["zenotravel_17"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task17.pddl.csv"
    instances["zenotravel_18"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task18.pddl.csv"
    instances["zenotravel_19"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task19.pddl.csv"
    instances["zenotravel_20"] = "data/preprocessing/"+expeirment_type+"/zenotravel/task20.pddl.csv"

    instances["airport_01"] = "data/preprocessing/"+expeirment_type+"/airport/task01.pddl.csv"
    instances["airport_02"] = "data/preprocessing/"+expeirment_type+"/airport/task02.pddl.csv"
    instances["airport_03"] = "data/preprocessing/"+expeirment_type+"/airport/task03.pddl.csv"
    instances["airport_04"] = "data/preprocessing/"+expeirment_type+"/airport/task04.pddl.csv"
    instances["airport_05"] = "data/preprocessing/"+expeirment_type+"/airport/task05.pddl.csv"
    instances["airport_06"] = "data/preprocessing/"+expeirment_type+"/airport/task06.pddl.csv"
    instances["airport_07"] = "data/preprocessing/"+expeirment_type+"/airport/task07.pddl.csv"
    instances["airport_08"] = "data/preprocessing/"+expeirment_type+"/airport/task08.pddl.csv"
    instances["airport_09"] = "data/preprocessing/"+expeirment_type+"/airport/task09.pddl.csv"
    instances["airport_10"] = "data/preprocessing/"+expeirment_type+"/airport/task10.pddl.csv"
    instances["airport_11"] = "data/preprocessing/"+expeirment_type+"/airport/task11.pddl.csv"
    instances["airport_12"] = "data/preprocessing/"+expeirment_type+"/airport/task12.pddl.csv"
    instances["airport_13"] = "data/preprocessing/"+expeirment_type+"/airport/task13.pddl.csv"
    instances["airport_14"] = "data/preprocessing/"+expeirment_type+"/airport/task14.pddl.csv"
    instances["airport_15"] = "data/preprocessing/"+expeirment_type+"/airport/task15.pddl.csv"
    instances["airport_16"] = "data/preprocessing/"+expeirment_type+"/airport/task16.pddl.csv"
    instances["airport_17"] = "data/preprocessing/"+expeirment_type+"/airport/task17.pddl.csv"
    instances["airport_18"] = "data/preprocessing/"+expeirment_type+"/airport/task18.pddl.csv"
    instances["airport_19"] = "data/preprocessing/"+expeirment_type+"/airport/task19.pddl.csv"
    instances["airport_20"] = "data/preprocessing/"+expeirment_type+"/airport/task20.pddl.csv"
    instances["airport_21"] = "data/preprocessing/"+expeirment_type+"/airport/task21.pddl.csv"
    instances["airport_22"] = "data/preprocessing/"+expeirment_type+"/airport/task22.pddl.csv"
    instances["airport_23"] = "data/preprocessing/"+expeirment_type+"/airport/task23.pddl.csv"
    instances["airport_24"] = "data/preprocessing/"+expeirment_type+"/airport/task24.pddl.csv"
    instances["airport_25"] = "data/preprocessing/"+expeirment_type+"/airport/task25.pddl.csv"
    instances["airport_26"] = "data/preprocessing/"+expeirment_type+"/airport/task26.pddl.csv"
    instances["airport_27"] = "data/preprocessing/"+expeirment_type+"/airport/task27.pddl.csv"
    instances["airport_28"] = "data/preprocessing/"+expeirment_type+"/airport/task28.pddl.csv"
    instances["airport_29"] = "data/preprocessing/"+expeirment_type+"/airport/task29.pddl.csv"
    instances["airport_30"] = "data/preprocessing/"+expeirment_type+"/airport/task30.pddl.csv"

    instances["blocks_01"] = "data/preprocessing/"+expeirment_type+"/blocks/task01.pddl.csv"
    instances["blocks_02"] = "data/preprocessing/"+expeirment_type+"/blocks/task02.pddl.csv"
    instances["blocks_03"] = "data/preprocessing/"+expeirment_type+"/blocks/task03.pddl.csv"
    instances["blocks_04"] = "data/preprocessing/"+expeirment_type+"/blocks/task04.pddl.csv"
    instances["blocks_05"] = "data/preprocessing/"+expeirment_type+"/blocks/task05.pddl.csv"
    instances["blocks_06"] = "data/preprocessing/"+expeirment_type+"/blocks/task06.pddl.csv"
    instances["blocks_07"] = "data/preprocessing/"+expeirment_type+"/blocks/task07.pddl.csv"
    instances["blocks_08"] = "data/preprocessing/"+expeirment_type+"/blocks/task08.pddl.csv"
    instances["blocks_09"] = "data/preprocessing/"+expeirment_type+"/blocks/task09.pddl.csv"
    instances["blocks_10"] = "data/preprocessing/"+expeirment_type+"/blocks/task10.pddl.csv"
    instances["blocks_11"] = "data/preprocessing/"+expeirment_type+"/blocks/task11.pddl.csv"
    instances["blocks_12"] = "data/preprocessing/"+expeirment_type+"/blocks/task12.pddl.csv"
    instances["blocks_13"] = "data/preprocessing/"+expeirment_type+"/blocks/task13.pddl.csv"
    instances["blocks_14"] = "data/preprocessing/"+expeirment_type+"/blocks/task14.pddl.csv"
    instances["blocks_15"] = "data/preprocessing/"+expeirment_type+"/blocks/task15.pddl.csv"
    instances["blocks_16"] = "data/preprocessing/"+expeirment_type+"/blocks/task16.pddl.csv"
    instances["blocks_17"] = "data/preprocessing/"+expeirment_type+"/blocks/task17.pddl.csv"
    instances["blocks_18"] = "data/preprocessing/"+expeirment_type+"/blocks/task18.pddl.csv"
    instances["blocks_19"] = "data/preprocessing/"+expeirment_type+"/blocks/task19.pddl.csv"
    instances["blocks_20"] = "data/preprocessing/"+expeirment_type+"/blocks/task20.pddl.csv"
    instances["blocks_21"] = "data/preprocessing/"+expeirment_type+"/blocks/task21.pddl.csv"
    instances["blocks_22"] = "data/preprocessing/"+expeirment_type+"/blocks/task22.pddl.csv"
    instances["blocks_23"] = "data/preprocessing/"+expeirment_type+"/blocks/task23.pddl.csv"
    instances["blocks_24"] = "data/preprocessing/"+expeirment_type+"/blocks/task24.pddl.csv"
    instances["blocks_25"] = "data/preprocessing/"+expeirment_type+"/blocks/task25.pddl.csv"
    instances["blocks_26"] = "data/preprocessing/"+expeirment_type+"/blocks/task26.pddl.csv"
    instances["blocks_27"] = "data/preprocessing/"+expeirment_type+"/blocks/task27.pddl.csv"
    instances["blocks_28"] = "data/preprocessing/"+expeirment_type+"/blocks/task28.pddl.csv"
    instances["blocks_29"] = "data/preprocessing/"+expeirment_type+"/blocks/task29.pddl.csv"
    instances["blocks_30"] = "data/preprocessing/"+expeirment_type+"/blocks/task30.pddl.csv"
    instances["blocks_31"] = "data/preprocessing/"+expeirment_type+"/blocks/task31.pddl.csv"
    instances["blocks_33"] = "data/preprocessing/"+expeirment_type+"/blocks/task33.pddl.csv"
    instances["blocks_34"] = "data/preprocessing/"+expeirment_type+"/blocks/task34.pddl.csv"
    instances["blocks_35"] = "data/preprocessing/"+expeirment_type+"/blocks/task35.pddl.csv"


    instances["depot_01"] = "data/preprocessing/"+expeirment_type+"/depot/task01.pddl.csv"
    instances["depot_02"] = "data/preprocessing/"+expeirment_type+"/depot/task02.pddl.csv"
    instances["depot_03"] = "data/preprocessing/"+expeirment_type+"/depot/task03.pddl.csv"
    instances["depot_04"] = "data/preprocessing/"+expeirment_type+"/depot/task04.pddl.csv"
    instances["depot_05"] = "data/preprocessing/"+expeirment_type+"/depot/task05.pddl.csv"
    instances["depot_06"] = "data/preprocessing/"+expeirment_type+"/depot/task06.pddl.csv"
    instances["depot_07"] = "data/preprocessing/"+expeirment_type+"/depot/task07.pddl.csv"
    instances["depot_08"] = "data/preprocessing/"+expeirment_type+"/depot/task08.pddl.csv"
    instances["depot_09"] = "data/preprocessing/"+expeirment_type+"/depot/task09.pddl.csv"
    instances["depot_10"] = "data/preprocessing/"+expeirment_type+"/depot/task10.pddl.csv"
    instances["depot_11"] = "data/preprocessing/"+expeirment_type+"/depot/task11.pddl.csv"
    instances["depot_12"] = "data/preprocessing/"+expeirment_type+"/depot/task12.pddl.csv"
    instances["depot_13"] = "data/preprocessing/"+expeirment_type+"/depot/task13.pddl.csv"
    instances["depot_14"] = "data/preprocessing/"+expeirment_type+"/depot/task14.pddl.csv"
    instances["depot_15"] = "data/preprocessing/"+expeirment_type+"/depot/task15.pddl.csv"
    instances["depot_16"] = "data/preprocessing/"+expeirment_type+"/depot/task16.pddl.csv"
    instances["depot_17"] = "data/preprocessing/"+expeirment_type+"/depot/task17.pddl.csv"
    instances["depot_18"] = "data/preprocessing/"+expeirment_type+"/depot/task18.pddl.csv"
    instances["depot_19"] = "data/preprocessing/"+expeirment_type+"/depot/task19.pddl.csv"
    instances["depot_20"] = "data/preprocessing/"+expeirment_type+"/depot/task20.pddl.csv"
    instances["depot_21"] = "data/preprocessing/"+expeirment_type+"/depot/task21.pddl.csv"
    instances["depot_22"] = "data/preprocessing/"+expeirment_type+"/depot/task22.pddl.csv"

    instances["elevators_01"] = "data/preprocessing/"+expeirment_type+"/elevators/task01.pddl.csv"
    instances["elevators_02"] = "data/preprocessing/"+expeirment_type+"/elevators/task02.pddl.csv"
    instances["elevators_03"] = "data/preprocessing/"+expeirment_type+"/elevators/task03.pddl.csv"
    instances["elevators_04"] = "data/preprocessing/"+expeirment_type+"/elevators/task04.pddl.csv"
    instances["elevators_05"] = "data/preprocessing/"+expeirment_type+"/elevators/task05.pddl.csv"
    instances["elevators_06"] = "data/preprocessing/"+expeirment_type+"/elevators/task06.pddl.csv"
    instances["elevators_07"] = "data/preprocessing/"+expeirment_type+"/elevators/task07.pddl.csv"
    instances["elevators_08"] = "data/preprocessing/"+expeirment_type+"/elevators/task08.pddl.csv"
    instances["elevators_09"] = "data/preprocessing/"+expeirment_type+"/elevators/task09.pddl.csv"
    instances["elevators_10"] = "data/preprocessing/"+expeirment_type+"/elevators/task10.pddl.csv"
    instances["elevators_11"] = "data/preprocessing/"+expeirment_type+"/elevators/task11.pddl.csv"
    instances["elevators_12"] = "data/preprocessing/"+expeirment_type+"/elevators/task12.pddl.csv"
    instances["elevators_13"] = "data/preprocessing/"+expeirment_type+"/elevators/task13.pddl.csv"
    instances["elevators_14"] = "data/preprocessing/"+expeirment_type+"/elevators/task14.pddl.csv"
    instances["elevators_15"] = "data/preprocessing/"+expeirment_type+"/elevators/task15.pddl.csv"
    instances["elevators_16"] = "data/preprocessing/"+expeirment_type+"/elevators/task16.pddl.csv"
    instances["elevators_17"] = "data/preprocessing/"+expeirment_type+"/elevators/task17.pddl.csv"
    instances["elevators_18"] = "data/preprocessing/"+expeirment_type+"/elevators/task18.pddl.csv"
    instances["elevators_19"] = "data/preprocessing/"+expeirment_type+"/elevators/task19.pddl.csv"
    instances["elevators_20"] = "data/preprocessing/"+expeirment_type+"/elevators/task20.pddl.csv"
    instances["elevators_21"] = "data/preprocessing/"+expeirment_type+"/elevators/task21.pddl.csv"
    instances["elevators_22"] = "data/preprocessing/"+expeirment_type+"/elevators/task22.pddl.csv"
    instances["elevators_23"] = "data/preprocessing/"+expeirment_type+"/elevators/task23.pddl.csv"
    instances["elevators_24"] = "data/preprocessing/"+expeirment_type+"/elevators/task24.pddl.csv"
    instances["elevators_25"] = "data/preprocessing/"+expeirment_type+"/elevators/task25.pddl.csv"
    instances["elevators_26"] = "data/preprocessing/"+expeirment_type+"/elevators/task26.pddl.csv"
    instances["elevators_27"] = "data/preprocessing/"+expeirment_type+"/elevators/task27.pddl.csv"
    instances["elevators_28"] = "data/preprocessing/"+expeirment_type+"/elevators/task28.pddl.csv"
    instances["elevators_29"] = "data/preprocessing/"+expeirment_type+"/elevators/task29.pddl.csv"
    instances["elevators_30"] = "data/preprocessing/"+expeirment_type+"/elevators/task30.pddl.csv"


    instances["freecell_01"] = "data/preprocessing/"+expeirment_type+"/freecell/task01.pddl.csv"
    instances["freecell_02"] = "data/preprocessing/"+expeirment_type+"/freecell/task02.pddl.csv"
    instances["freecell_03"] = "data/preprocessing/"+expeirment_type+"/freecell/task03.pddl.csv"
    instances["freecell_04"] = "data/preprocessing/"+expeirment_type+"/freecell/task04.pddl.csv"
    instances["freecell_05"] = "data/preprocessing/"+expeirment_type+"/freecell/task05.pddl.csv"
    instances["freecell_06"] = "data/preprocessing/"+expeirment_type+"/freecell/task06.pddl.csv"
    instances["freecell_07"] = "data/preprocessing/"+expeirment_type+"/freecell/task07.pddl.csv"
    instances["freecell_08"] = "data/preprocessing/"+expeirment_type+"/freecell/task08.pddl.csv"
    instances["freecell_09"] = "data/preprocessing/"+expeirment_type+"/freecell/task09.pddl.csv"
    instances["freecell_10"] = "data/preprocessing/"+expeirment_type+"/freecell/task10.pddl.csv"
    instances["freecell_11"] = "data/preprocessing/"+expeirment_type+"/freecell/task11.pddl.csv"
    instances["freecell_12"] = "data/preprocessing/"+expeirment_type+"/freecell/task12.pddl.csv"
    instances["freecell_13"] = "data/preprocessing/"+expeirment_type+"/freecell/task13.pddl.csv"
    instances["freecell_14"] = "data/preprocessing/"+expeirment_type+"/freecell/task14.pddl.csv"
    instances["freecell_15"] = "data/preprocessing/"+expeirment_type+"/freecell/task15.pddl.csv"
    instances["freecell_16"] = "data/preprocessing/"+expeirment_type+"/freecell/task16.pddl.csv"
    instances["freecell_17"] = "data/preprocessing/"+expeirment_type+"/freecell/task17.pddl.csv"
    instances["freecell_18"] = "data/preprocessing/"+expeirment_type+"/freecell/task18.pddl.csv"
    instances["freecell_19"] = "data/preprocessing/"+expeirment_type+"/freecell/task19.pddl.csv"
    instances["freecell_20"] = "data/preprocessing/"+expeirment_type+"/freecell/task20.pddl.csv"

    instances["gripper_01"] = "data/preprocessing/"+expeirment_type+"/gripper/task01.pddl.csv"
    instances["gripper_02"] = "data/preprocessing/"+expeirment_type+"/gripper/task02.pddl.csv"
    instances["gripper_03"] = "data/preprocessing/"+expeirment_type+"/gripper/task03.pddl.csv"
    instances["gripper_04"] = "data/preprocessing/"+expeirment_type+"/gripper/task04.pddl.csv"
    instances["gripper_05"] = "data/preprocessing/"+expeirment_type+"/gripper/task05.pddl.csv"
    instances["gripper_06"] = "data/preprocessing/"+expeirment_type+"/gripper/task06.pddl.csv"
    instances["gripper_07"] = "data/preprocessing/"+expeirment_type+"/gripper/task07.pddl.csv"
    instances["gripper_08"] = "data/preprocessing/"+expeirment_type+"/gripper/task08.pddl.csv"
    instances["gripper_09"] = "data/preprocessing/"+expeirment_type+"/gripper/task09.pddl.csv"
    instances["gripper_10"] = "data/preprocessing/"+expeirment_type+"/gripper/task10.pddl.csv"
    instances["gripper_11"] = "data/preprocessing/"+expeirment_type+"/gripper/task11.pddl.csv"
    instances["gripper_12"] = "data/preprocessing/"+expeirment_type+"/gripper/task12.pddl.csv"
    instances["gripper_13"] = "data/preprocessing/"+expeirment_type+"/gripper/task13.pddl.csv"
    instances["gripper_14"] = "data/preprocessing/"+expeirment_type+"/gripper/task14.pddl.csv"
    instances["gripper_15"] = "data/preprocessing/"+expeirment_type+"/gripper/task15.pddl.csv"
    instances["gripper_16"] = "data/preprocessing/"+expeirment_type+"/gripper/task16.pddl.csv"
    instances["gripper_17"] = "data/preprocessing/"+expeirment_type+"/gripper/task17.pddl.csv"
    instances["gripper_18"] = "data/preprocessing/"+expeirment_type+"/gripper/task18.pddl.csv"
    instances["gripper_19"] = "data/preprocessing/"+expeirment_type+"/gripper/task19.pddl.csv"
    instances["gripper_20"] = "data/preprocessing/"+expeirment_type+"/gripper/task20.pddl.csv"

    instances["logistics_01"] = "data/preprocessing/"+expeirment_type+"/logistics/task01.pddl.csv"
    instances["logistics_02"] = "data/preprocessing/"+expeirment_type+"/logistics/task02.pddl.csv"
    instances["logistics_03"] = "data/preprocessing/"+expeirment_type+"/logistics/task03.pddl.csv"
    instances["logistics_04"] = "data/preprocessing/"+expeirment_type+"/logistics/task04.pddl.csv"
    instances["logistics_05"] = "data/preprocessing/"+expeirment_type+"/logistics/task05.pddl.csv"
    instances["logistics_06"] = "data/preprocessing/"+expeirment_type+"/logistics/task06.pddl.csv"
    instances["logistics_07"] = "data/preprocessing/"+expeirment_type+"/logistics/task07.pddl.csv"
    instances["logistics_08"] = "data/preprocessing/"+expeirment_type+"/logistics/task08.pddl.csv"
    instances["logistics_09"] = "data/preprocessing/"+expeirment_type+"/logistics/task09.pddl.csv"
    instances["logistics_10"] = "data/preprocessing/"+expeirment_type+"/logistics/task10.pddl.csv"
    instances["logistics_11"] = "data/preprocessing/"+expeirment_type+"/logistics/task11.pddl.csv"
    instances["logistics_12"] = "data/preprocessing/"+expeirment_type+"/logistics/task12.pddl.csv"
    instances["logistics_13"] = "data/preprocessing/"+expeirment_type+"/logistics/task13.pddl.csv"
    instances["logistics_14"] = "data/preprocessing/"+expeirment_type+"/logistics/task14.pddl.csv"
    instances["logistics_15"] = "data/preprocessing/"+expeirment_type+"/logistics/task15.pddl.csv"
    instances["logistics_16"] = "data/preprocessing/"+expeirment_type+"/logistics/task16.pddl.csv"
    instances["logistics_17"] = "data/preprocessing/"+expeirment_type+"/logistics/task17.pddl.csv"
    instances["logistics_18"] = "data/preprocessing/"+expeirment_type+"/logistics/task18.pddl.csv"
    instances["logistics_19"] = "data/preprocessing/"+expeirment_type+"/logistics/task19.pddl.csv"
    instances["logistics_20"] = "data/preprocessing/"+expeirment_type+"/logistics/task20.pddl.csv"
    instances["logistics_21"] = "data/preprocessing/"+expeirment_type+"/logistics/task21.pddl.csv"
    instances["logistics_22"] = "data/preprocessing/"+expeirment_type+"/logistics/task22.pddl.csv"
    instances["logistics_23"] = "data/preprocessing/"+expeirment_type+"/logistics/task23.pddl.csv"
    instances["logistics_24"] = "data/preprocessing/"+expeirment_type+"/logistics/task24.pddl.csv"
    instances["logistics_25"] = "data/preprocessing/"+expeirment_type+"/logistics/task25.pddl.csv"
    instances["logistics_26"] = "data/preprocessing/"+expeirment_type+"/logistics/task26.pddl.csv"
    instances["logistics_27"] = "data/preprocessing/"+expeirment_type+"/logistics/task27.pddl.csv"
    instances["logistics_28"] = "data/preprocessing/"+expeirment_type+"/logistics/task28.pddl.csv"

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

    instances["movie_01"] = "data/preprocessing/"+expeirment_type+"/movie/task01.pddl.csv"
    instances["movie_02"] = "data/preprocessing/"+expeirment_type+"/movie/task02.pddl.csv"
    instances["movie_03"] = "data/preprocessing/"+expeirment_type+"/movie/task03.pddl.csv"
    instances["movie_04"] = "data/preprocessing/"+expeirment_type+"/movie/task04.pddl.csv"
    instances["movie_05"] = "data/preprocessing/"+expeirment_type+"/movie/task05.pddl.csv"
    instances["movie_06"] = "data/preprocessing/"+expeirment_type+"/movie/task06.pddl.csv"
    instances["movie_07"] = "data/preprocessing/"+expeirment_type+"/movie/task07.pddl.csv"
    instances["movie_08"] = "data/preprocessing/"+expeirment_type+"/movie/task08.pddl.csv"
    instances["movie_09"] = "data/preprocessing/"+expeirment_type+"/movie/task09.pddl.csv"
    instances["movie_10"] = "data/preprocessing/"+expeirment_type+"/movie/task10.pddl.csv"
    instances["movie_11"] = "data/preprocessing/"+expeirment_type+"/movie/task11.pddl.csv"
    instances["movie_12"] = "data/preprocessing/"+expeirment_type+"/movie/task12.pddl.csv"
    instances["movie_13"] = "data/preprocessing/"+expeirment_type+"/movie/task13.pddl.csv"
    instances["movie_14"] = "data/preprocessing/"+expeirment_type+"/movie/task14.pddl.csv"
    instances["movie_15"] = "data/preprocessing/"+expeirment_type+"/movie/task15.pddl.csv"
    instances["movie_16"] = "data/preprocessing/"+expeirment_type+"/movie/task16.pddl.csv"
    instances["movie_17"] = "data/preprocessing/"+expeirment_type+"/movie/task17.pddl.csv"
    instances["movie_18"] = "data/preprocessing/"+expeirment_type+"/movie/task18.pddl.csv"
    instances["movie_19"] = "data/preprocessing/"+expeirment_type+"/movie/task19.pddl.csv"
    instances["movie_20"] = "data/preprocessing/"+expeirment_type+"/movie/task20.pddl.csv"
    instances["movie_21"] = "data/preprocessing/"+expeirment_type+"/movie/task21.pddl.csv"
    instances["movie_22"] = "data/preprocessing/"+expeirment_type+"/movie/task22.pddl.csv"
    instances["movie_23"] = "data/preprocessing/"+expeirment_type+"/movie/task23.pddl.csv"
    instances["movie_24"] = "data/preprocessing/"+expeirment_type+"/movie/task24.pddl.csv"
    instances["movie_25"] = "data/preprocessing/"+expeirment_type+"/movie/task25.pddl.csv"
    instances["movie_26"] = "data/preprocessing/"+expeirment_type+"/movie/task26.pddl.csv"
    instances["movie_27"] = "data/preprocessing/"+expeirment_type+"/movie/task27.pddl.csv"
    instances["movie_28"] = "data/preprocessing/"+expeirment_type+"/movie/task28.pddl.csv"
    instances["movie_29"] = "data/preprocessing/"+expeirment_type+"/movie/task29.pddl.csv"
    instances["movie_30"] = "data/preprocessing/"+expeirment_type+"/movie/task30.pddl.csv"


    instances["openstacks_01"] = "data/preprocessing/"+expeirment_type+"/openstacks/task01.pddl.csv"
    instances["openstacks_02"] = "data/preprocessing/"+expeirment_type+"/openstacks/task02.pddl.csv"
    instances["openstacks_03"] = "data/preprocessing/"+expeirment_type+"/openstacks/task03.pddl.csv"
    instances["openstacks_04"] = "data/preprocessing/"+expeirment_type+"/openstacks/task04.pddl.csv"
    instances["openstacks_05"] = "data/preprocessing/"+expeirment_type+"/openstacks/task05.pddl.csv"
    instances["openstacks_06"] = "data/preprocessing/"+expeirment_type+"/openstacks/task06.pddl.csv"
    instances["openstacks_07"] = "data/preprocessing/"+expeirment_type+"/openstacks/task07.pddl.csv"
    instances["openstacks_08"] = "data/preprocessing/"+expeirment_type+"/openstacks/task08.pddl.csv"
    instances["openstacks_09"] = "data/preprocessing/"+expeirment_type+"/openstacks/task09.pddl.csv"
    instances["openstacks_10"] = "data/preprocessing/"+expeirment_type+"/openstacks/task10.pddl.csv"
    instances["openstacks_11"] = "data/preprocessing/"+expeirment_type+"/openstacks/task12.pddl.csv"
    instances["openstacks_13"] = "data/preprocessing/"+expeirment_type+"/openstacks/task13.pddl.csv"
    instances["openstacks_14"] = "data/preprocessing/"+expeirment_type+"/openstacks/task14.pddl.csv"
    instances["openstacks_15"] = "data/preprocessing/"+expeirment_type+"/openstacks/task15.pddl.csv"
    instances["openstacks_16"] = "data/preprocessing/"+expeirment_type+"/openstacks/task16.pddl.csv"
    instances["openstacks_17"] = "data/preprocessing/"+expeirment_type+"/openstacks/task17.pddl.csv"
    instances["openstacks_18"] = "data/preprocessing/"+expeirment_type+"/openstacks/task18.pddl.csv"
    instances["openstacks_19"] = "data/preprocessing/"+expeirment_type+"/openstacks/task19.pddl.csv"
    instances["openstacks_20"] = "data/preprocessing/"+expeirment_type+"/openstacks/task20.pddl.csv"
    instances["openstacks_21"] = "data/preprocessing/"+expeirment_type+"/openstacks/task21.pddl.csv"
    instances["openstacks_22"] = "data/preprocessing/"+expeirment_type+"/openstacks/task22.pddl.csv"
    instances["openstacks_23"] = "data/preprocessing/"+expeirment_type+"/openstacks/task23.pddl.csv"
    instances["openstacks_24"] = "data/preprocessing/"+expeirment_type+"/openstacks/task24.pddl.csv"
    instances["openstacks_25"] = "data/preprocessing/"+expeirment_type+"/openstacks/task25.pddl.csv"
    instances["openstacks_26"] = "data/preprocessing/"+expeirment_type+"/openstacks/task26.pddl.csv"
    instances["openstacks_27"] = "data/preprocessing/"+expeirment_type+"/openstacks/task27.pddl.csv"
    instances["openstacks_28"] = "data/preprocessing/"+expeirment_type+"/openstacks/task28.pddl.csv"
    instances["openstacks_29"] = "data/preprocessing/"+expeirment_type+"/openstacks/task29.pddl.csv"
    instances["openstacks_30"] = "data/preprocessing/"+expeirment_type+"/openstacks/task30.pddl.csv"

    instances["parcprinter_01"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task01.pddl.csv"
    instances["parcprinter_02"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task02.pddl.csv"
    instances["parcprinter_03"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task03.pddl.csv"
    instances["parcprinter_04"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task04.pddl.csv"
    instances["parcprinter_05"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task05.pddl.csv"
    instances["parcprinter_06"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task06.pddl.csv"
    instances["parcprinter_07"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task07.pddl.csv"
    instances["parcprinter_08"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task08.pddl.csv"
    instances["parcprinter_09"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task09.pddl.csv"
    instances["parcprinter_10"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task10.pddl.csv"
    instances["parcprinter_11"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task11.pddl.csv"
    instances["parcprinter_12"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task12.pddl.csv"
    instances["parcprinter_13"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task13.pddl.csv"
    instances["parcprinter_14"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task14.pddl.csv"
    instances["parcprinter_15"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task15.pddl.csv"
    instances["parcprinter_16"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task16.pddl.csv"
    instances["parcprinter_17"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task17.pddl.csv"
    instances["parcprinter_18"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task18.pddl.csv"
    instances["parcprinter_19"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task19.pddl.csv"
    instances["parcprinter_20"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task20.pddl.csv"
    instances["parcprinter_21"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task21.pddl.csv"
    instances["parcprinter_22"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task22.pddl.csv"
    instances["parcprinter_23"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task23.pddl.csv"
    instances["parcprinter_24"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task24.pddl.csv"
    instances["parcprinter_25"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task25.pddl.csv"
    instances["parcprinter_26"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task26.pddl.csv"
    instances["parcprinter_27"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task27.pddl.csv"
    instances["parcprinter_28"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task28.pddl.csv"
    instances["parcprinter_29"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task29.pddl.csv"
    instances["parcprinter_30"] = "data/preprocessing/"+expeirment_type+"/parcprinter/task30.pddl.csv"

    instances["pegsol_01"] = "data/preprocessing/"+expeirment_type+"/pegsol/task01.pddl.csv"
    instances["pegsol_02"] = "data/preprocessing/"+expeirment_type+"/pegsol/task02.pddl.csv"
    instances["pegsol_03"] = "data/preprocessing/"+expeirment_type+"/pegsol/task03.pddl.csv"
    instances["pegsol_04"] = "data/preprocessing/"+expeirment_type+"/pegsol/task04.pddl.csv"
    instances["pegsol_05"] = "data/preprocessing/"+expeirment_type+"/pegsol/task05.pddl.csv"
    instances["pegsol_06"] = "data/preprocessing/"+expeirment_type+"/pegsol/task06.pddl.csv"
    instances["pegsol_07"] = "data/preprocessing/"+expeirment_type+"/pegsol/task07.pddl.csv"
    instances["pegsol_08"] = "data/preprocessing/"+expeirment_type+"/pegsol/task08.pddl.csv"
    instances["pegsol_09"] = "data/preprocessing/"+expeirment_type+"/pegsol/task09.pddl.csv"
    instances["pegsol_10"] = "data/preprocessing/"+expeirment_type+"/pegsol/task10.pddl.csv"
    instances["pegsol_11"] = "data/preprocessing/"+expeirment_type+"/pegsol/task11.pddl.csv"
    instances["pegsol_12"] = "data/preprocessing/"+expeirment_type+"/pegsol/task12.pddl.csv"
    instances["pegsol_13"] = "data/preprocessing/"+expeirment_type+"/pegsol/task13.pddl.csv"
    instances["pegsol_14"] = "data/preprocessing/"+expeirment_type+"/pegsol/task14.pddl.csv"
    instances["pegsol_15"] = "data/preprocessing/"+expeirment_type+"/pegsol/task15.pddl.csv"
    instances["pegsol_16"] = "data/preprocessing/"+expeirment_type+"/pegsol/task16.pddl.csv"
    instances["pegsol_17"] = "data/preprocessing/"+expeirment_type+"/pegsol/task17.pddl.csv"
    instances["pegsol_18"] = "data/preprocessing/"+expeirment_type+"/pegsol/task18.pddl.csv"
    instances["pegsol_19"] = "data/preprocessing/"+expeirment_type+"/pegsol/task19.pddl.csv"
    instances["pegsol_20"] = "data/preprocessing/"+expeirment_type+"/pegsol/task20.pddl.csv"
    instances["pegsol_21"] = "data/preprocessing/"+expeirment_type+"/pegsol/task21.pddl.csv"
    instances["pegsol_22"] = "data/preprocessing/"+expeirment_type+"/pegsol/task22.pddl.csv"
    instances["pegsol_23"] = "data/preprocessing/"+expeirment_type+"/pegsol/task23.pddl.csv"
    instances["pegsol_24"] = "data/preprocessing/"+expeirment_type+"/pegsol/task24.pddl.csv"
    instances["pegsol_25"] = "data/preprocessing/"+expeirment_type+"/pegsol/task25.pddl.csv"
    instances["pegsol_26"] = "data/preprocessing/"+expeirment_type+"/pegsol/task26.pddl.csv"
    instances["pegsol_27"] = "data/preprocessing/"+expeirment_type+"/pegsol/task27.pddl.csv"
    instances["pegsol_28"] = "data/preprocessing/"+expeirment_type+"/pegsol/task28.pddl.csv"
    instances["pegsol_29"] = "data/preprocessing/"+expeirment_type+"/pegsol/task29.pddl.csv"
    instances["pegsol_30"] = "data/preprocessing/"+expeirment_type+"/pegsol/task30.pddl.csv"

    instances["psr-small_01"] = "data/preprocessing/"+expeirment_type+"/psr-small/task01.pddl.csv"
    instances["psr-small_02"] = "data/preprocessing/"+expeirment_type+"/psr-small/task02.pddl.csv"
    instances["psr-small_03"] = "data/preprocessing/"+expeirment_type+"/psr-small/task03.pddl.csv"
    instances["psr-small_04"] = "data/preprocessing/"+expeirment_type+"/psr-small/task04.pddl.csv"
    instances["psr-small_05"] = "data/preprocessing/"+expeirment_type+"/psr-small/task05.pddl.csv"
    instances["psr-small_06"] = "data/preprocessing/"+expeirment_type+"/psr-small/task06.pddl.csv"
    instances["psr-small_07"] = "data/preprocessing/"+expeirment_type+"/psr-small/task07.pddl.csv"
    instances["psr-small_08"] = "data/preprocessing/"+expeirment_type+"/psr-small/task08.pddl.csv"
    instances["psr-small_09"] = "data/preprocessing/"+expeirment_type+"/psr-small/task09.pddl.csv"
    instances["psr-small_10"] = "data/preprocessing/"+expeirment_type+"/psr-small/task10.pddl.csv"
    instances["psr-small_11"] = "data/preprocessing/"+expeirment_type+"/psr-small/task11.pddl.csv"
    instances["psr-small_12"] = "data/preprocessing/"+expeirment_type+"/psr-small/task12.pddl.csv"
    instances["psr-small_13"] = "data/preprocessing/"+expeirment_type+"/psr-small/task13.pddl.csv"
    instances["psr-small_14"] = "data/preprocessing/"+expeirment_type+"/psr-small/task14.pddl.csv"
    instances["psr-small_15"] = "data/preprocessing/"+expeirment_type+"/psr-small/task15.pddl.csv"
    instances["psr-small_16"] = "data/preprocessing/"+expeirment_type+"/psr-small/task16.pddl.csv"
    instances["psr-small_17"] = "data/preprocessing/"+expeirment_type+"/psr-small/task17.pddl.csv"
    instances["psr-small_18"] = "data/preprocessing/"+expeirment_type+"/psr-small/task18.pddl.csv"
    instances["psr-small_19"] = "data/preprocessing/"+expeirment_type+"/psr-small/task19.pddl.csv"
    instances["psr-small_20"] = "data/preprocessing/"+expeirment_type+"/psr-small/task20.pddl.csv"
    instances["psr-small_21"] = "data/preprocessing/"+expeirment_type+"/psr-small/task21.pddl.csv"
    instances["psr-small_22"] = "data/preprocessing/"+expeirment_type+"/psr-small/task22.pddl.csv"
    instances["psr-small_23"] = "data/preprocessing/"+expeirment_type+"/psr-small/task23.pddl.csv"
    instances["psr-small_24"] = "data/preprocessing/"+expeirment_type+"/psr-small/task24.pddl.csv"
    instances["psr-small_25"] = "data/preprocessing/"+expeirment_type+"/psr-small/task25.pddl.csv"
    instances["psr-small_26"] = "data/preprocessing/"+expeirment_type+"/psr-small/task26.pddl.csv"
    instances["psr-small_27"] = "data/preprocessing/"+expeirment_type+"/psr-small/task27.pddl.csv"
    instances["psr-small_28"] = "data/preprocessing/"+expeirment_type+"/psr-small/task28.pddl.csv"
    instances["psr-small_29"] = "data/preprocessing/"+expeirment_type+"/psr-small/task29.pddl.csv"
    instances["psr-small_30"] = "data/preprocessing/"+expeirment_type+"/psr-small/task30.pddl.csv"
    instances["psr-small_31"] = "data/preprocessing/"+expeirment_type+"/psr-small/task31.pddl.csv"
    instances["psr-small_32"] = "data/preprocessing/"+expeirment_type+"/psr-small/task32.pddl.csv"
    instances["psr-small_33"] = "data/preprocessing/"+expeirment_type+"/psr-small/task33.pddl.csv"
    instances["psr-small_34"] = "data/preprocessing/"+expeirment_type+"/psr-small/task34.pddl.csv"
    instances["psr-small_35"] = "data/preprocessing/"+expeirment_type+"/psr-small/task35.pddl.csv"
    instances["psr-small_36"] = "data/preprocessing/"+expeirment_type+"/psr-small/task36.pddl.csv"
    instances["psr-small_37"] = "data/preprocessing/"+expeirment_type+"/psr-small/task37.pddl.csv"
    instances["psr-small_38"] = "data/preprocessing/"+expeirment_type+"/psr-small/task38.pddl.csv"
    instances["psr-small_39"] = "data/preprocessing/"+expeirment_type+"/psr-small/task39.pddl.csv"
    instances["psr-small_40"] = "data/preprocessing/"+expeirment_type+"/psr-small/task40.pddl.csv"
    instances["psr-small_41"] = "data/preprocessing/"+expeirment_type+"/psr-small/task41.pddl.csv"
    instances["psr-small_42"] = "data/preprocessing/"+expeirment_type+"/psr-small/task42.pddl.csv"
    instances["psr-small_43"] = "data/preprocessing/"+expeirment_type+"/psr-small/task43.pddl.csv"
    instances["psr-small_44"] = "data/preprocessing/"+expeirment_type+"/psr-small/task44.pddl.csv"
    instances["psr-small_45"] = "data/preprocessing/"+expeirment_type+"/psr-small/task45.pddl.csv"
    instances["psr-small_46"] = "data/preprocessing/"+expeirment_type+"/psr-small/task46.pddl.csv"
    instances["psr-small_47"] = "data/preprocessing/"+expeirment_type+"/psr-small/task47.pddl.csv"
    instances["psr-small_48"] = "data/preprocessing/"+expeirment_type+"/psr-small/task48.pddl.csv"
    instances["psr-small_49"] = "data/preprocessing/"+expeirment_type+"/psr-small/task49.pddl.csv"
    instances["psr-small_50"] = "data/preprocessing/"+expeirment_type+"/psr-small/task50.pddl.csv"

    instances["rovers_01"] = "data/preprocessing/"+expeirment_type+"/rovers/task01.pddl.csv"
    instances["rovers_02"] = "data/preprocessing/"+expeirment_type+"/rovers/task02.pddl.csv"
    instances["rovers_03"] = "data/preprocessing/"+expeirment_type+"/rovers/task03.pddl.csv"
    instances["rovers_04"] = "data/preprocessing/"+expeirment_type+"/rovers/task04.pddl.csv"
    instances["rovers_05"] = "data/preprocessing/"+expeirment_type+"/rovers/task05.pddl.csv"
    instances["rovers_06"] = "data/preprocessing/"+expeirment_type+"/rovers/task06.pddl.csv"
    instances["rovers_07"] = "data/preprocessing/"+expeirment_type+"/rovers/task07.pddl.csv"
    instances["rovers_08"] = "data/preprocessing/"+expeirment_type+"/rovers/task08.pddl.csv"
    instances["rovers_09"] = "data/preprocessing/"+expeirment_type+"/rovers/task09.pddl.csv"
    instances["rovers_10"] = "data/preprocessing/"+expeirment_type+"/rovers/task10.pddl.csv"
    instances["rovers_11"] = "data/preprocessing/"+expeirment_type+"/rovers/task11.pddl.csv"
    instances["rovers_12"] = "data/preprocessing/"+expeirment_type+"/rovers/task12.pddl.csv"
    instances["rovers_13"] = "data/preprocessing/"+expeirment_type+"/rovers/task13.pddl.csv"
    instances["rovers_14"] = "data/preprocessing/"+expeirment_type+"/rovers/task14.pddl.csv"
    instances["rovers_15"] = "data/preprocessing/"+expeirment_type+"/rovers/task15.pddl.csv"
    instances["rovers_16"] = "data/preprocessing/"+expeirment_type+"/rovers/task16.pddl.csv"
    instances["rovers_17"] = "data/preprocessing/"+expeirment_type+"/rovers/task17.pddl.csv"
    instances["rovers_18"] = "data/preprocessing/"+expeirment_type+"/rovers/task18.pddl.csv"
    instances["rovers_19"] = "data/preprocessing/"+expeirment_type+"/rovers/task19.pddl.csv"
    instances["rovers_20"] = "data/preprocessing/"+expeirment_type+"/rovers/task20.pddl.csv"
    instances["rovers_21"] = "data/preprocessing/"+expeirment_type+"/rovers/task21.pddl.csv"
    instances["rovers_22"] = "data/preprocessing/"+expeirment_type+"/rovers/task22.pddl.csv"
    instances["rovers_23"] = "data/preprocessing/"+expeirment_type+"/rovers/task23.pddl.csv"
    instances["rovers_24"] = "data/preprocessing/"+expeirment_type+"/rovers/task24.pddl.csv"
    instances["rovers_25"] = "data/preprocessing/"+expeirment_type+"/rovers/task25.pddl.csv"
    instances["rovers_26"] = "data/preprocessing/"+expeirment_type+"/rovers/task26.pddl.csv"
    instances["rovers_27"] = "data/preprocessing/"+expeirment_type+"/rovers/task27.pddl.csv"
    instances["rovers_28"] = "data/preprocessing/"+expeirment_type+"/rovers/task28.pddl.csv"
    instances["rovers_29"] = "data/preprocessing/"+expeirment_type+"/rovers/task29.pddl.csv"
    instances["rovers_30"] = "data/preprocessing/"+expeirment_type+"/rovers/task30.pddl.csv"

    instances["satellite_01"] = "data/preprocessing/"+expeirment_type+"/satellite/task01.pddl.csv"
    instances["satellite_02"] = "data/preprocessing/"+expeirment_type+"/satellite/task02.pddl.csv"
    instances["satellite_03"] = "data/preprocessing/"+expeirment_type+"/satellite/task03.pddl.csv"
    instances["satellite_04"] = "data/preprocessing/"+expeirment_type+"/satellite/task04.pddl.csv"
    instances["satellite_05"] = "data/preprocessing/"+expeirment_type+"/satellite/task05.pddl.csv"
    instances["satellite_06"] = "data/preprocessing/"+expeirment_type+"/satellite/task06.pddl.csv"
    instances["satellite_07"] = "data/preprocessing/"+expeirment_type+"/satellite/task07.pddl.csv"
    instances["satellite_08"] = "data/preprocessing/"+expeirment_type+"/satellite/task08.pddl.csv"
    instances["satellite_09"] = "data/preprocessing/"+expeirment_type+"/satellite/task09.pddl.csv"
    instances["satellite_10"] = "data/preprocessing/"+expeirment_type+"/satellite/task10.pddl.csv"
    instances["satellite_11"] = "data/preprocessing/"+expeirment_type+"/satellite/task11.pddl.csv"
    instances["satellite_12"] = "data/preprocessing/"+expeirment_type+"/satellite/task12.pddl.csv"
    instances["satellite_13"] = "data/preprocessing/"+expeirment_type+"/satellite/task13.pddl.csv"
    instances["satellite_14"] = "data/preprocessing/"+expeirment_type+"/satellite/task14.pddl.csv"
    instances["satellite_15"] = "data/preprocessing/"+expeirment_type+"/satellite/task15.pddl.csv"
    instances["satellite_16"] = "data/preprocessing/"+expeirment_type+"/satellite/task16.pddl.csv"
    instances["satellite_17"] = "data/preprocessing/"+expeirment_type+"/satellite/task17.pddl.csv"
    instances["satellite_18"] = "data/preprocessing/"+expeirment_type+"/satellite/task18.pddl.csv"
    instances["satellite_19"] = "data/preprocessing/"+expeirment_type+"/satellite/task19.pddl.csv"
    instances["satellite_20"] = "data/preprocessing/"+expeirment_type+"/satellite/task20.pddl.csv"


    instances["scanalyzer_01"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task01.pddl.csv"
    instances["scanalyzer_02"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task02.pddl.csv"
    instances["scanalyzer_03"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task03.pddl.csv"
    instances["scanalyzer_04"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task04.pddl.csv"
    instances["scanalyzer_05"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task05.pddl.csv"
    instances["scanalyzer_06"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task06.pddl.csv"
    instances["scanalyzer_07"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task07.pddl.csv"
    instances["scanalyzer_08"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task08.pddl.csv"
    instances["scanalyzer_09"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task09.pddl.csv"
    instances["scanalyzer_10"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task10.pddl.csv"
    instances["scanalyzer_11"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task11.pddl.csv"
    instances["scanalyzer_12"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task12.pddl.csv"
    instances["scanalyzer_13"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task13.pddl.csv"
    instances["scanalyzer_14"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task14.pddl.csv"
    instances["scanalyzer_15"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task15.pddl.csv"
    instances["scanalyzer_16"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task16.pddl.csv"
    instances["scanalyzer_17"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task17.pddl.csv"
    instances["scanalyzer_18"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task18.pddl.csv"
    instances["scanalyzer_19"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task19.pddl.csv"
    instances["scanalyzer_20"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task20.pddl.csv"
    instances["scanalyzer_21"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task21.pddl.csv"
    instances["scanalyzer_22"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task22.pddl.csv"
    instances["scanalyzer_23"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task23.pddl.csv"
    instances["scanalyzer_24"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task24.pddl.csv"
    instances["scanalyzer_25"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task25.pddl.csv"
    instances["scanalyzer_26"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task26.pddl.csv"
    instances["scanalyzer_27"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task27.pddl.csv"
    instances["scanalyzer_28"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task28.pddl.csv"
    instances["scanalyzer_29"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task29.pddl.csv"
    instances["scanalyzer_30"] = "data/preprocessing/"+expeirment_type+"/scanalyzer/task30.pddl.csv"


    instances["sokoban_01"] = "data/preprocessing/"+expeirment_type+"/sokoban/task01.pddl.csv"
    instances["sokoban_02"] = "data/preprocessing/"+expeirment_type+"/sokoban/task02.pddl.csv"
    instances["sokoban_03"] = "data/preprocessing/"+expeirment_type+"/sokoban/task03.pddl.csv"
    instances["sokoban_04"] = "data/preprocessing/"+expeirment_type+"/sokoban/task04.pddl.csv"
    instances["sokoban_05"] = "data/preprocessing/"+expeirment_type+"/sokoban/task05.pddl.csv"
    instances["sokoban_06"] = "data/preprocessing/"+expeirment_type+"/sokoban/task06.pddl.csv"
    instances["sokoban_07"] = "data/preprocessing/"+expeirment_type+"/sokoban/task07.pddl.csv"
    instances["sokoban_08"] = "data/preprocessing/"+expeirment_type+"/sokoban/task08.pddl.csv"
    instances["sokoban_09"] = "data/preprocessing/"+expeirment_type+"/sokoban/task09.pddl.csv"
    instances["sokoban_10"] = "data/preprocessing/"+expeirment_type+"/sokoban/task10.pddl.csv"
    instances["sokoban_11"] = "data/preprocessing/"+expeirment_type+"/sokoban/task11.pddl.csv"
    instances["sokoban_12"] = "data/preprocessing/"+expeirment_type+"/sokoban/task12.pddl.csv"
    instances["sokoban_13"] = "data/preprocessing/"+expeirment_type+"/sokoban/task13.pddl.csv"
    instances["sokoban_14"] = "data/preprocessing/"+expeirment_type+"/sokoban/task14.pddl.csv"
    instances["sokoban_15"] = "data/preprocessing/"+expeirment_type+"/sokoban/task15.pddl.csv"
    instances["sokoban_16"] = "data/preprocessing/"+expeirment_type+"/sokoban/task16.pddl.csv"
    instances["sokoban_17"] = "data/preprocessing/"+expeirment_type+"/sokoban/task17.pddl.csv"
    instances["sokoban_18"] = "data/preprocessing/"+expeirment_type+"/sokoban/task18.pddl.csv"
    instances["sokoban_19"] = "data/preprocessing/"+expeirment_type+"/sokoban/task19.pddl.csv"
    instances["sokoban_20"] = "data/preprocessing/"+expeirment_type+"/sokoban/task12.pddl.csv"
    instances["sokoban_21"] = "data/preprocessing/"+expeirment_type+"/sokoban/task21.pddl.csv"
    instances["sokoban_22"] = "data/preprocessing/"+expeirment_type+"/sokoban/task22.pddl.csv"
    instances["sokoban_23"] = "data/preprocessing/"+expeirment_type+"/sokoban/task23.pddl.csv"
    instances["sokoban_24"] = "data/preprocessing/"+expeirment_type+"/sokoban/task24.pddl.csv"
    instances["sokoban_25"] = "data/preprocessing/"+expeirment_type+"/sokoban/task25.pddl.csv"
    instances["sokoban_26"] = "data/preprocessing/"+expeirment_type+"/sokoban/task26.pddl.csv"
    instances["sokoban_27"] = "data/preprocessing/"+expeirment_type+"/sokoban/task27.pddl.csv"
    instances["sokoban_28"] = "data/preprocessing/"+expeirment_type+"/sokoban/task28.pddl.csv"
    instances["sokoban_29"] = "data/preprocessing/"+expeirment_type+"/sokoban/task29.pddl.csv"
    instances["sokoban_30"] = "data/preprocessing/"+expeirment_type+"/sokoban/task30.pddl.csv"
    
    original_path = os.getcwd()
    K=30
    num_of_featuers = len(features_columns)
    for instance in instances:
        #checking if we already did the file or we are missing the output
        if os.path.isfile("temp/"+instance+"_x_.pt") or os.path.isfile(instances[instance]) == False:
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

