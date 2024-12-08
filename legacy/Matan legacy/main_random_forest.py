import torch
import hydra
from train import train, evaluate
from dataset import MyDataset
from models.base_model import LSTM
from torch.utils.data import DataLoader
#from utils import main_utils, train_utils, preprocessing_dataset
from utils import main_utils, train_utils, preprocessing_dataset
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
#from utils.main_utils import tensor_to_list
from utils.main_utils import tensor_to_list, tensor_to_list_splitted
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd

torch.backends.cudnn.benchmark = True

def get_folder_name(part_name):
    for directories in os.listdir(os.getcwd()+"/logs/"):
        if part_name in directories:
            return directories
    raise

@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
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
    metric = "mae"
    result_to_csv =[]
    for current_k in [1]: #[15, 25]:
        for planner in ["gbfs_hff", "astar_hff", "astar_lmcut", "gbfs_lmcut"]:
            for domain in domains:
                cfg['main']['experiment_name_prefix'] = str(current_k)+"_"+planner+"_"+domain #+odd_or_even
                main_utils.init(cfg)
                logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
                logger.write(OmegaConf.to_yaml(cfg))

                # Set seed for results reproduction
                main_utils.set_seed(cfg['main']['seed'])

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                print("device = ", device)

                features_columns = ["node#", "BF", "node_f", "node_h", "node_g" , "father_node#", "father_BF", "father_node_f", "father_node_h",
                                    "father_node_g", "grandfather_node#", "grandfather_BF", "grandfather_node_f","grandfather_node_h",
                                    "grandfather_node_g"]
                num_features = len(features_columns)
                labels_columns = ["y_norm"] 
                K=cfg['main']['K']
                train_name = cfg['train']['train_name']
                test_name = cfg['train']['test_name']
                preprocessing_path = cfg['main']['paths']['preprocessing_path']
                data_path = cfg['main']['paths']['data_path']

                x_train, y_train, x_test, y_test = preprocessing_dataset.prepare_dataset_random_forest(
                    full_dataset=cfg['main']['full_dataset'], folder_path=data_path,
                    current_k=current_k, planner=planner, domain=domain
                    )
                if (x_train == None or x_test == None):
                    continue
                x_train = x_train.view(x_train.shape[0], x_train.shape[2])
                for index in range(len(x_test)):
                    x_test[index] = x_test[index].view(x_test[index].shape[0], x_test[index].shape[2])
                x_test, y_test = tensor_to_list_splitted(x_test, y_test)
                x_train, y_train = tensor_to_list_splitted(x_train, y_train)
                
                #test_dataset = tensor_to_list_splitted(x_test, y_test)
                #x_train = y_train = x_test = y_test = None

                #train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                                        #num_workers=cfg['main']['num_workers'])
                #eval_loader = DataLoader(test_dataset, cfg['train']['batch_size'], shuffle=False,
                                        #num_workers=cfg['main']['num_workers'])


                #X, y = make_regression(n_features=19, n_informative=2,random_state=0, shuffle=True)
                regr = RandomForestRegressor(max_depth=19, random_state=0)
                regr.fit(x_train, y_train)
                for index in range(len(x_test)):
                    current_x_test = x_test[index]
                    current_y_test = y_test[index]
                    output = regr.predict(current_x_test)
                    if metric == "mse":
                        diff = (output-current_y_test) * (output-current_y_test)
                    else:
                        diff = abs(output-current_y_test)
                    sum = diff.sum()/len(current_y_test)
                    result_to_csv.append((sum, str(current_k)+"_"+planner+"_"+domain+"_"+str(index)))
    x_df = pd.DataFrame(result_to_csv)
    x_df.to_csv("all_scores_random_forest_mae.csv")

if __name__ == '__main__':
    main()




