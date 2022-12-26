import yaml
import dataclasses

@dataclasses.dataclass(frozen=True)
class Config:
    root_dir : str
    dataset_dir : str
    model_name : str
    test_extract : bool
    fps_sampling : int
    fps_sampling_test : int
    batch_size : int
    num_tasks : int
    num_workers : int
    num_sanity_val_steps : int
    input_size : int
    out_features : int
    tool_features : int
    features_subsampling : int
    features_per_seconds : int
    learning_rate : float
    early_stopping_metric : str
    pretrained : bool
    save_top_k : int
    max_epocks : int
    min_epocks : int
    gpus : list
    feature_output_path : str
    name : str
    wandb : bool
    train : bool
    features_per_seconds : int
    features_subsampling : int
    log_every_n_steps : int
    add_tool_feats : bool
    model_path : str
    
@dataclasses.dataclass(frozen=True)
class Config_ibfocal:
    root_dir : str
    dataset_dir : str
    model_name : str
    test_extract : bool
    fps_sampling : int
    fps_sampling_test : int
    batch_size : int
    num_tasks : int
    num_workers : int
    num_sanity_val_steps : int
    input_size : int
    out_features : int
    tool_features : int
    features_subsampling : int
    features_per_seconds : int
    learning_rate : float
    early_stopping_metric : str
    pretrained : bool
    save_top_k : int
    ib_start_epoch : int
    max_epocks : int
    min_epocks : int
    gpus : list
    feature_output_path : str
    wandb : bool
    train : bool
    name : str
    model_path : str
    log_every_n_steps : int
    add_tool_feats : bool


@dataclasses.dataclass(frozen=True)
class Config_tecno:
    root_dir : str
    batch_size : int  
    gpus : list
    out_features : int
    num_workers : int
    save_top_k : int
    num_sanity_val_steps : int
    early_stopping_metric : str
    feature_output_path : str
    features_name : str
    features_per_seconds : int
    features_subsampling : int
    mstcn_output_path : str
    log_every_n_steps : int
    mstcn_causal_conv : bool
    mstcn_learning_rate : float
    mstcn_min_epochs : int
    mstcn_max_epochs : int
    mstcn_layers : int
    mstcn_f_maps : int
    mstcn_f_dim: int
    mstcn_stages : int
    mstcn_early_stopping_metric : str
    wandb : bool
    train : bool
    model_path : str
    name : str
    
def get_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config
    
def get_config_tecno(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config_tecno(**config_dict)
    return config
    
def get_config_ibfocal(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config_ibfocal(**config_dict)
    return config