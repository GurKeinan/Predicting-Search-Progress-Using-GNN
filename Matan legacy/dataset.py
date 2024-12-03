import torch
from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List

class MyDataset(Dataset):
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
        self.y = pd.read_csv(labels_path_data, skip_blank_lines=True, header=0, usecols=labels_columns)
        self.len = len(self.y)
        self.features_columns = features_columns
        self.labels_columns = labels_columns

    def Dataset(self, K):
        temp_padding = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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
            temp_list_labels.append(self.y[self.labels_columns][i:(i+1)].values[0][0])
            main_list.append((temp_list_features.copy(), temp_list_labels.copy()))
            temp_list_labels.pop(0)
        return main_list   
    
    def final_dataset(self, K):
        main_list = self.Dataset(K)
        n_instances = len(main_list)
        x = torch.zeros((n_instances, K, 15))
        y = torch.zeros(n_instances)
        for i in range(n_instances):
            x[i] = torch.tensor(main_list[i][0])
            y[i] = torch.tensor(main_list[i][1])
        
        return (x,y)
    
    def __len__(self):
        return len(self.df[:])