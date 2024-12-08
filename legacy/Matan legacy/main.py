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
from utils.main_utils import tensor_to_list
import os
import time

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


    for current_k in [5,3]: #[15, 25]:
        for planner in ["gbfs_hff", "astar_hff", "astar_lmcut", "gbfs_lmcut"]:
            for regime in ["OD"]:#, "SD", "ODTS"]:
                #if (planner == "gbfs_hff" and regime == "OD"):
                    #continue
                odd_or_even_options = ["train"]
                if regime == "SD" or regime == "ODTS":
                    odd_or_even_options.append("test")
                    cfg['train']['batch_size'] = 128
                else:
                    cfg['train']['batch_size'] = 1024
                for odd_or_even in odd_or_even_options:
                    for domain in domains:
                        cfg['main']['experiment_name_prefix'] = str(current_k)+"_"+planner+"_"+regime+"_"+domain+"_"+odd_or_even #+odd_or_even
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

                        x_train, y_train, x_test, y_test = preprocessing_dataset.prepare_dataset(
                            full_dataset=cfg['main']['full_dataset'], folder_path=data_path,
                            current_k=current_k, planner=planner, regime=regime, domain=domain,
                            odd_or_even=odd_or_even
                            )
                        if (x_train == None or x_test == None):
                            continue
                        train_dataset = tensor_to_list(x_train, y_train)
                        test_dataset = tensor_to_list(x_test, y_test)
                        x_train = y_train = x_test = y_test = None

                        train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                                                num_workers=cfg['main']['num_workers'])
                        eval_loader = DataLoader(test_dataset, cfg['train']['batch_size'], shuffle=False,
                                                num_workers=cfg['main']['num_workers'])

                        # Init model
                        model = LSTM(
                            input_size=cfg['train']['input_size'],
                            hidden_size=cfg['train']['hidden_size'],
                            num_layers=cfg['train']['hidden_size'],
                            K=current_k, #cfg['main']['K'],
                            p_dropout=cfg['train']['p_dropout'],
                            bias=cfg['train']['bias'],
                            bidirectional=cfg['train']['bidirectional'],
                            )


                        #if cfg['main']['load_model']:
                        if regime == "ODTS":
                            path = get_folder_name(str(current_k)+"_"+planner+"_OD_"+domain+"_train")
                            #temp = torch.load("logs/gbfs_lmcut/exp_3_part_16_tpp/model.pth")
                            #temp = torch.load(os.getcwd()+"/logs/new_exp/"+path+"/model.pth")
                            temp = torch.load(os.getcwd()+"/logs/"+path+"/model.pth")
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

                        logger.write(main_utils.get_model_string(model))

                        # Run model
                        train_params = train_utils.get_train_params(cfg)

                        # Report metrics and hyper parameters to tensorboard
                        if cfg['main']['run_test_only']:
                            start = time.time()
                            print(evaluate(model, eval_loader, 0, False, 20, None, cfg['main']["metric"]))
                            end = time.time()
                        else:
                            metrics = train(model, train_loader, eval_loader, train_params, logger, cfg['main']["metric"])
                            hyper_parameters = main_utils.get_flatten_dict(cfg['train'])
                            logger.report_metrics_hyper_params(hyper_parameters, metrics)

if __name__ == '__main__':
    main()




