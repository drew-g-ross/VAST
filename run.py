import os 
import json
import torch
import torch.distributed as dist
from vast_utils.args import get_args,logging_cfgs
from vast_utils.initialize import initialize
from vast_utils.build_model import build_model
from vast_utils.build_optimizer import build_optimizer 
from vast_utils.build_dataloader import create_train_dataloaders, create_val_dataloaders
from vast_utils.pipeline import train, test

# UNCOMMENT FOR DEBUG / NOT USING torch.distributed
# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '0'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '9834'

def main():

    ### init 

    args = get_args()
    initialize(args)

    ### logging cfgs
    logging_cfgs(args)   


    if args.run_cfg.mode == 'training':

        ### create datasets and dataloader
        train_loader = create_train_dataloaders(args)
        val_loaders = create_val_dataloaders(args)

        ### build model and optimizer

        model, optimizer_ckpt, start_step = build_model(args)

        optimizer = build_optimizer(model, args, optimizer_ckpt)


        ### start evaluation
        # if args.run_cfg.first_eval or args.run_cfg.zero_shot:
        #     test(model, val_loaders, args.run_cfg)  
        #     print("SUCCESSFULLY TESTED")                               
        #     if args.run_cfg.zero_shot:
        #         return 

        ### start training


        train(model, optimizer, train_loader, val_loaders, args.run_cfg, start_step = start_step, verbose_time=False)

    elif args.run_cfg.mode == 'testing':
        ### build model
        model,_,_ = build_model(args)

        ### create datasets and dataloader
        val_loaders = create_val_dataloaders(args)

        ### start evaluation
        test(model, val_loaders, args.run_cfg)                                 

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
