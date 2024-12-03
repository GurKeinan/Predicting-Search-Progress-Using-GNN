import torch
import hydra
from train import train, evaluate
from models.base_model import LSTM
from torch.utils.data import DataLoader
from utils import main_utils, train_utils, preprocessing_dataset
from omegaconf import DictConfig, OmegaConf
from utils.main_utils import tensor_to_list
import os
import pandas as pd
import tqdm

torch.backends.cudnn.benchmark = True

def get_folder_name(part_name):
    for directories in os.listdir(os.getcwd()+"/logs/new_exp/"):
        if part_name in directories:
            return directories
    return None

@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    cfg['main']['load_model'] = True
    cfg['main']['run_test_only'] = True
    cfg['main']['full_dataset'] = True
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
    for current_k in [40]:
        for train_planner in tqdm.tqdm(["astar_hff", "astar_lmcut", "gbfs_lmcut", "gbfs_hff"]):
            for test_planner in tqdm.tqdm(["astar_hff", "astar_lmcut", "gbfs_lmcut", "gbfs_hff"]):
                result_to_csv =[]
                for regime in ["OD"]:
                    odd_or_even_options = ["train"]
                    cfg['train']['batch_size'] = 1024
                    for odd_or_even in odd_or_even_options:
                        for domain in domains:
                            cfg['main']['experiment_name_prefix'] = str(current_k)+"_"+train_planner+"_"+regime+"_"+domain+"_"+odd_or_even #+odd_or_even
                            main_utils.init(cfg)
                            cfg['main']['load_model'] = True
                            cfg['main']['run_test_only'] = True
                            cfg['main']['full_dataset'] = True
                            main_utils.set_seed(cfg['main']['seed'])

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

                            x_train, y_train, x_test, y_test = preprocessing_dataset.prepare_dataset2(
                                full_dataset=cfg['main']['full_dataset'], folder_path=data_path,
                                current_k=current_k, planner=test_planner, regime=regime, domain=domain,
                                odd_or_even=odd_or_even
                                )
                            if (len(x_test) < 1 or len(x_train) < 1):
                                continue


                            for index in range(len(x_test)):
                                current_x_test = x_test[index]
                                current_y_test = y_test[index]
                                test_dataset = tensor_to_list(current_x_test, current_y_test)
                                eval_loader = DataLoader(test_dataset, cfg['train']['batch_size'], shuffle=False,
                                                        num_workers=cfg['main']['num_workers'])

                                # Init model
                                model = LSTM(
                                    input_size=cfg['train']['input_size'],
                                    hidden_size=cfg['train']['hidden_size'],
                                    num_layers=cfg['train']['hidden_size'],
                                    K=current_k,
                                    p_dropout=cfg['train']['p_dropout'],
                                    bias=cfg['train']['bias'],
                                    bidirectional=cfg['train']['bidirectional'],
                                    )


                                if cfg['main']['load_model']:
                                    path = get_folder_name(str(current_k)+"_"+train_planner+"_"+regime+"_"+domain+"_"+odd_or_even)
                                    if (path == None):
                                        continue
                                    if (os.path.isfile(os.getcwd()+"/logs/new_exp/"+path+"/model.pth")==False):
                                        continue
                                    temp = torch.load(os.getcwd()+"/logs/new_exp/"+path+"/model.pth")
                                    torch.save(temp['model_state'], 'tensor.pt')
                                    temp2 = torch.load('tensor.pt')
                                    temp3 = {}
                                    for i in temp2:
                                        new_i = i.replace('module.', '')
                                        temp3[new_i] = temp2[i]
                                    model.load_state_dict(temp3)

                                # Add gpus_to_use
                                if cfg['main']['parallel']:
                                    model = torch.nn.DataParallel(model)

                                if torch.cuda.is_available():
                                    model = model.cuda()


                                # Run model
                                train_params = train_utils.get_train_params(cfg)

                                # Report metrics and hyper parameters to tensorboard
                                if cfg['main']['run_test_only']:
                                    score, loss = evaluate(model, eval_loader, 0, False, 20, None, cfg['main']["metric"])
                                    result_to_csv.append((-score, str(current_k)+"_train_"+train_planner+"_test_"+test_planner+"_"+regime+"_"+domain+"_"+odd_or_even+str(index)))
                x_df = pd.DataFrame(result_to_csv)
                x_df.to_csv("cross_domains_"+train_planner+"_"+test_planner+".csv")
if __name__ == '__main__':
    main()




