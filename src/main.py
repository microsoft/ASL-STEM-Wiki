import os
import pickle

import torch

import hydra
from omegaconf import OmegaConf

from train import *
from heuristic_eval import *

@hydra.main(config_path='../hydra_configs', config_name='main', version_base=None)
def main(cfg):
    print('Config:', cfg)
    
    # add runtime info to cfg
    OmegaConf.set_struct(cfg, False)
    cfg.meta = OmegaConf.create({})
    cfg.meta.original_dir = hydra.utils.get_original_cwd()
    cfg.meta.run_dir = os.getcwd()
    cfg.meta.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Meta-config:', cfg.meta)
    
    #####
    
    results = {}

    if cfg.model.type == 'heuristic':
        run_results = run_heuristic_eval(cfg)
    else:
        run_results = train(cfg)
    if cfg.verbose:
        print('Run results:', run_results)
    
    results['loss'] = run_results
    
    #####

    results['cfg'] = OmegaConf.to_container(cfg)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
