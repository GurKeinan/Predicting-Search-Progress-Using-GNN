"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'K': int,
        'version': str,
        'save': bool,
        'load_model': bool,
        'run_test_only': bool,
        'full_dataset': bool,
        'metric': str,
        'paths': {
            'logs': str,
            'data_path': str,
            'versions_dir': str,
            'preprocessing_path': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'train_name': str,
        'test_name': str,
        'hidden_size': int,
        'num_layers': int,
        'p_dropout': float,
        'bias': bool,
        'input_size': int,
        'print_plots': bool,
        'print_plot_every_epochs': int,
        'bidirectional': bool,
        'lr': {
            'lr_value': float,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
}
