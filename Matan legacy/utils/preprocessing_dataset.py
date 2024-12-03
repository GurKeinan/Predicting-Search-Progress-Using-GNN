import os
from utils import *
from torch.autograd import Variable
import torchviz
import tqdm

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

def prepare_dataset(full_dataset, folder_path, current_k, planner, regime, domain, odd_or_even):

    domains = [
        "airport",
        "blocks",
        "depot",
        "elevators",
        "freecell",
        "gripper",
        "logistics",
        "miconic",
        "movie",
        "openstacks",
        "parcprinter",
        "pegsol",
        "psr-small",
        "rovers",
        "satellite",
        "scanalyzer",
        "sokoban",
        "tpp",
        "transport",
        "woodworking",
        "zenotravel"
    ]
    instances = {}
    for current_domain in domains:
        for instance_index in range(1,51):
            str_instance = str(instance_index)
            if (len(str_instance) == 1):
                str_instance = "0"+str_instance
            full_path = folder_path+str(current_k)+"/"+planner+"/"+current_domain+"_"+str_instance+"_x_.pt"
            if os.path.isfile(os.getcwd()+"/"+full_path) == False:
                continue
            if (regime == "OD"):
                if current_domain == domain:
                    instances[current_domain+"_"+str_instance] = "test"
                else:
                    instances[current_domain+"_"+str_instance] = "train"
            elif (regime == "SD" or regime == "ODTS"):
                if current_domain == domain:
                    instances[current_domain+"_"+str_instance] = odd_or_even
                    if odd_or_even == "train":
                        odd_or_even = "test"
                    else:
                        odd_or_even = "train"
    path = folder_path+str(current_k)+"/"+planner+"/"
    number_of_nodes_in_each_instance = 1000
    train_x = None
    train_y = None
    test_x = None
    test_y = None
    for instance in tqdm.tqdm(instances):
        temp_x = torch.load(path+instance+"_x_.pt")
        temp_y = torch.load(path+instance+"_y_.pt")
        temp_x = temp_x[:-1]
        temp_y = temp_y[:-1]
        if (instances[instance] == "train"):
            #if this is the 1st instance
            if (train_x == None):
                if full_dataset == False:
                    temp_x, temp_y = select_k_nodes(temp_x, temp_y,  number_of_nodes_in_each_instance)
                train_x = temp_x
                train_y = temp_y
            else:
                if full_dataset == False:
                    temp_x, temp_y = select_k_nodes(temp_x, temp_y,  number_of_nodes_in_each_instance)
                train_x = torch.cat((train_x, temp_x), 0)
                train_y = torch.cat((train_y, temp_y), 0)
        else:
            temp_x = torch.load(path+instance+"_x_.pt")
            temp_y = torch.load(path+instance+"_y_.pt")
            temp_x = temp_x[:-1]
            temp_y = temp_y[:-1]
            #if this is the 1st instance
            if (test_x == None):
                if full_dataset == False:
                    temp_x, temp_y = select_k_nodes(temp_x, temp_y,  number_of_nodes_in_each_instance)
                test_x = temp_x
                test_y = temp_y
            else:
                if full_dataset == False:
                    temp_x, temp_y = select_k_nodes(temp_x, temp_y,  number_of_nodes_in_each_instance)
                test_x = torch.cat((test_x, temp_x), 0)
                test_y = torch.cat((test_y, temp_y), 0)
    return (train_x, train_y, test_x, test_y)


def prepare_dataset2(full_dataset, folder_path, current_k, planner, regime, domain, odd_or_even):

    domains = [
        "airport",
        "blocks",
        "depot",
        "elevators",
        "freecell",
        "gripper",
        "logistics",
        "miconic",
        "movie",
        "openstacks",
        "parcprinter",
        "pegsol",
        "psr-small",
        "rovers",
        "satellite",
        "scanalyzer",
        "sokoban",
        "tpp",
        "transport",
        "woodworking",
        "zenotravel"
    ]
    instances = {}
    for current_domain in domains:
        for instance_index in range(1,51):
            str_instance = str(instance_index)
            if (len(str_instance) == 1):
                str_instance = "0"+str_instance
            # 'data/ready_data/'
            full_path = folder_path+str(current_k)+"/"+planner+"/"+current_domain+"_"+str_instance+"_x_.pt"
            if os.path.isfile(os.getcwd()+"/"+full_path) == False:
                continue
            if (regime == "OD"):
                if current_domain == domain:
                    instances[current_domain+"_"+str_instance] = "test"
                else:
                    instances[current_domain+"_"+str_instance] = "train"
            elif (regime == "SD" or regime == "ODTS"):
                if current_domain == domain:
                    instances[current_domain+"_"+str_instance] = odd_or_even
                    if odd_or_even == "train":
                        odd_or_even = "test"
                    else:
                        odd_or_even = "train"
    path = folder_path+str(current_k)+"/"+planner+"/"
    number_of_nodes_in_each_instance = 1000
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for instance in instances:
        print(path+instance+"_x_.pt")
        temp_x = torch.load(path+instance+"_x_.pt")
        temp_y = torch.load(path+instance+"_y_.pt")
        temp_x = temp_x[:-1]
        temp_y = temp_y[:-1]
        if (instances[instance] == "train"):
            train_x = [1]
        else:
            temp_x = torch.load(path+instance+"_x_.pt")
            temp_y = torch.load(path+instance+"_y_.pt")
            temp_x = temp_x[:-1]
            temp_y = temp_y[:-1]
            test_x.append(temp_x)
            test_y.append(temp_y)
    return (train_x, train_y, test_x, test_y)


def prepare_dataset_random_forest(full_dataset, folder_path, current_k, planner, domain):

    domains = [
        "airport",
        "blocks",
        "depot",
        "elevators",
        "freecell",
        "gripper",
        "logistics",
        "miconic",
        "movie",
        "openstacks",
        "parcprinter",
        "pegsol",
        "psr-small",
        "rovers",
        "satellite",
        "scanalyzer",
        "sokoban",
        "tpp",
        "transport",
        "woodworking",
        "zenotravel"
    ]
    instances = {}
    for current_domain in domains:
        for instance_index in range(1,51):
            str_instance = str(instance_index)
            if (len(str_instance) == 1):
                str_instance = "0"+str_instance
            full_path = folder_path+str(current_k)+"/"+planner+"/"+current_domain+"_"+str_instance+"_x_.pt"
            if os.path.isfile(os.getcwd()+"/"+full_path) == False:
                continue
            if current_domain == domain:
                instances[current_domain+"_"+str_instance] = "test"
            else:
                instances[current_domain+"_"+str_instance] = "train"
    path = folder_path+str(current_k)+"/"+planner+"/"
    number_of_nodes_in_each_instance = 1000
    train_x = None
    train_y = None
    test_x = []
    test_y = []
    for instance in instances:
        temp_x = torch.load(path+instance+"_x_.pt")
        temp_y = torch.load(path+instance+"_y_.pt")
        temp_x = temp_x[:-1]
        temp_y = temp_y[:-1]
        if (instances[instance] == "train"):
            #if this is the 1st instance
            if (train_x == None):
                if full_dataset == False:
                    temp_x, temp_y = select_k_nodes(temp_x, temp_y,  number_of_nodes_in_each_instance)
                train_x = temp_x
                train_y = temp_y
            else:
                if full_dataset == False:
                    temp_x, temp_y = select_k_nodes(temp_x, temp_y,  number_of_nodes_in_each_instance)
                train_x = torch.cat((train_x, temp_x), 0)
                train_y = torch.cat((train_y, temp_y), 0)
        else:
            temp_x = torch.load(path+instance+"_x_.pt")
            temp_y = torch.load(path+instance+"_y_.pt")
            temp_x = temp_x[:-1]
            temp_y = temp_y[:-1]
            test_x.append(temp_x)
            test_y.append(temp_y)
    return (train_x, train_y, test_x, test_y)

