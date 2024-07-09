import os
import torch
import torch.distributed as dist
from easydict import EasyDict as edict 

from model import model_registry
from torch.nn.parallel import DistributedDataParallel as DDP
from .logger import LOGGER
from .build_optimizer import build_optimizer


class DDP_modify(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module,name)

def build_model_fn(pretrain_dir):
    """Default pretrained model config that can be called functionally w/o command line"""
    model_cfg = edict({'model_type': 'vast',
                 'itm_ratio': 0.1,
                 'frozen_vision': False,
                 'frozen_audio': False,
                 'checkpointing': True,
                 'max_caption_len': 40,
                 'max_omni_caption_len': 70,
                 'max_subtitle_len': 70,
                 'contra_dim': 512,
                 'inherit_keys': ['vision_encoder_type', 'audio_encoder_type', 'audio_melbins', 'audio_target_length'],
                 'frame_embedding_type': 'adaptive',
                 'vision_resolution': 224,
                 'vision_encoder_type': 'evaclip01_giant',
                 'audio_encoder_type': 'beats',
                 'audio_melbins': 64,
                 'audio_target_length': 1024,
                 'beam_size': 3,
                 'captioner_mode': False,
                 'generate_nums': 1,
                 'ret_bidirection_evaluation': False,
                 'itm_rerank_num': 50,
                 'evaluation_type': 'evaluation_mm',
                 'default': './config/vast/default_model_cfg.json',
                 'max_vision_sample_num': 8,
                 'max_audio_sample_num': 1})
    model = model_registry[model_cfg.model_type](model_cfg)
    
    # load ckpt from a pretrained_dir
    args = edict({'run_cfg': {'pretrain_dir': pretrain_dir}})
    checkpoint = load_from_pretrained_dir(args)
    LOGGER.info("Load from pretrained dir {}".format(args.run_cfg.pretrain_dir))

    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    
    checkpoint = model.modify_checkpoint(checkpoint)
    if "model" in checkpoint.keys():
        checkpoint = checkpoint["model"]

    missing_keys,unexpected_keys = model.load_state_dict(checkpoint,strict=False)
    LOGGER.info(f"Unexpected keys {unexpected_keys}")
    LOGGER.info(f"missing_keys  {missing_keys}")
     
    # put on gpu
    local_rank = 0
    device = torch.device("cuda", local_rank)
    model.to(device)

    return model

def build_model(args):
    model = model_registry[args.model_cfg.model_type](args.model_cfg)
    checkpoint = {}
    
    ### load ckpt from a pretrained_dir
    if args.run_cfg.pretrain_dir:
        checkpoint = load_from_pretrained_dir(args)
        LOGGER.info("Load from pretrained dir {}".format(args.run_cfg.pretrain_dir))

    ### load ckpt from specific path
    if args.run_cfg.checkpoint:
        checkpoint =  torch.load(args.run_cfg.checkpoint, map_location = 'cpu')

    ### resume training
    if args.run_cfg.resume:
        checkpoint, checkpoint_optim, start_step = load_from_resume(args.run_cfg)
    else:
        checkpoint_optim, start_step = None , 0


    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    
    if checkpoint != {}:

        checkpoint = model.modify_checkpoint(checkpoint)
        if "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]

        missing_keys,unexpected_keys = model.load_state_dict(checkpoint,strict=False)
        LOGGER.info(f"Unexpected keys {unexpected_keys}")
        LOGGER.info(f"missing_keys  {missing_keys}")
     

    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    model.to(device)
    if  args.run_cfg.use_ddp:
        model = DDP_modify(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        pass

    return model, checkpoint_optim, start_step



def load_from_pretrained_dir(args):


    try: ### huggingface trainer
        checkpoint_dir = args.run_cfg.pretrain_dir
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('checkpoint')]
        checkpoint_ls = [int(i.split('-')[1]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
        
        try:
            checkpoint_name = f'checkpoint-{step}/pytorch_model.bin'
            ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
            checkpoint = torch.load(ckpt_file, map_location = 'cpu')
        except:
            checkpoint_name1 = f'checkpoint-{step}/pytorch_model-00001-of-00002.bin'
            ckpt_file1 = torch.load(os.path.join(checkpoint_dir, checkpoint_name1), map_location = 'cpu')
            checkpoint_name2 = f'checkpoint-{step}/pytorch_model-00002-of-00002.bin'
            ckpt_file2 = torch.load(os.path.join(checkpoint_dir, checkpoint_name2), map_location = 'cpu')
            ckpt_file1.update(ckpt_file2)
            checkpoint = ckpt_file1
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        LOGGER.info(f'load_from_pretrained: {ckpt_file}')

    except:
        checkpoint_dir = os.path.join(args.run_cfg.pretrain_dir,'ckpt')
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
            
        checkpoint_name = 'model_step_'+str(step)+'.pt'
        ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(ckpt_file, map_location = 'cpu')
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        LOGGER.info(f'load_from_pretrained: {ckpt_file}')


    return checkpoint


def load_from_resume(run_cfg):
    ckpt_dir = os.path.join(run_cfg.output_dir,'ckpt')
    previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
    steps = [i.split('.pt')[0].split('_')[-1] for i in  previous_optimizer_state] 
    steps = [ int(i) for i in steps]
    steps.sort()
    previous_step = steps[-1]
    previous_optimizer_state = f'optimizer_step_{previous_step}.pt'
    previous_model_state = f'model_step_{previous_step}.pt'
    previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
    previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
    previous_model_state = os.path.join(ckpt_dir, previous_model_state)
    
    assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
    LOGGER.info("choose previous model: {}".format(previous_model_state))
    LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    previous_model_state = torch.load(previous_model_state,map_location='cpu')
    previous_optimizer_state = torch.load(previous_optimizer_state,map_location='cpu')
    return previous_model_state, previous_optimizer_state, previous_step






