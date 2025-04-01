import os
import sys
import torch
import inspect
import copy
import argparse
import wandb
import matplotlib as plt

import config
from models.utils import get_model
from optim.base import train_base
from models.dg_mamba_llm import DGMambaLMHeadModel


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='markov', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def get_exp_name(args):
    """ Returns the name of the experiment, used for saving models and wandb. """
    exp_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}x{args.acc_steps}_{args.world_size}nodes"
    if args.wandb_run_prefix != 'none':
        exp_name = args.wandb_run_prefix + '_' + exp_name
    if 'sparse' in args.model:
        exp_name += f"_lmd{args.lmbda}"
    if getattr(args, 'dg', False):  # Add DG to experiment name if enabled
        exp_name += f"_dg{args.suppress_lambda}-{args.suppress_threshold}-{args.min_active_states}"
    exp_name += f"_seed={args.seed}"
    return exp_name


def main(args):
    type = args.type
    order = args.order
    generator = torch.Generator(device=args.device)
    generator.seed()

    # Markov transition probabilities (binary alphabet)
    if args.chain == "switch":
        if order == 1:
            p = args.p # 0... -> 1
            q = args.q # 1... -> 0
            P = torch.Tensor([[1-p, p],[q, 1-q]]).to(args.dtype).to(args.device)
        else:
            P = torch.zeros(2**order, 2, dtype=args.dtype, device=args.device)
            for k in range(2**order):
                pk = torch.rand(1, generator=generator, dtype=args.dtype, device=args.device)
                P[k,:] = torch.Tensor([1-pk, pk])
    elif args.chain == "random-fixed":
        P = torch.zeros(2**order, 2, dtype=args.dtype, device=args.device)
        for k in range(2**order):
            pk = torch.rand(1, generator=generator, dtype=args.dtype, device=args.device)
            P[k,:] = torch.Tensor([1-pk, pk])
    else:
        P = None

    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    torch.set_default_dtype(args.dtype)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)
    
    print(f"Loading dataset '{args.dataset}'")
    #dgmamba-enable
    if getattr(args, 'dg', False):
        print("Using DG-Mamba model")
        model = DGMambaLMHeadModel(args).to(args.device)
        model.configure_dg(
            enabled=True,
            suppress_lambda=args.suppress_lambda,
            suppress_threshold=args.suppress_threshold,
            min_active_states=args.min_active_states,
            current_epoch=0
        )
    else:
        model = get_model(args).to(args.device)
    
    group_specs = model.get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = [p_name]
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt/1e6,))
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    else:
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                                                            pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=.05)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    args.world_size = 1
    exp_name = get_exp_name(args)
    if args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy)
    
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")): # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    if args.model == 'base': # all train functions have the same interface
        train = train_base
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"\nTraining model={args.model} \n{vars(args)}\n")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    train(model, opt, P, type, order, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, generator,
        eval_freq=args.eval_freq, 
        ckpt_path=f"{ckpt_path}/ckpt.pt", extra_args=args)
    
    torch.save(model.state_dict(), 'model.pt')


if __name__ == "__main__":
    args = get_args()
    main(args)
