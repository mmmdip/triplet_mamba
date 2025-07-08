import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
# from transformers.optimization import AdamW
# import transformers
import matplotlib.pyplot as plt

from dataset import Dataset
from dataset_pretrain import PretrainDataset
from evaluator import Evaluator
from evaluator_pretrain import PretrainEvaluator
from modeling_gru import GRU_TS
from modeling_grud import GRUD_TS
from modeling_interpnet import InterpNet
from modeling_sand import SAND
from modeling_strats import Strats
from modeling_ehrmamba import EHR_Mamba
from modeling_duett import DuETT
from modeling_tripletmamba import Triplet_Mamba
from modeling_tcn import TCN_TS
from models import count_parameters
from utils import Logger, set_all_seeds


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--dataset', type=str, default='physionet_2012')
    parser.add_argument('--train_frac', type=float, default=0.5)
    parser.add_argument('--run', type=str, default='1o10')

    # model related arguments
    parser.add_argument('--model_type', type=str, default='strats',
                        choices=['gru', 'tcn', 'sand', 'grud', 'interpnet',
                                 'strats', 'istrats', 'ehrmamba', 'duett', 'tripletmamba'])
    parser.add_argument('--load_ckpt_path', type=str, default=None)
    ##  strats and istrats
    parser.add_argument('--max_obs', type=int, default=880)
    # tripletmamba related
    parser.add_argument('--cm_type', type=str, default='EinFFT')
    parser.add_argument('--sr_ratio', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    # parser.add_argument('--max_obs', type=int, default=8654)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)
    ## gru: hid_dim, dropout
    ## tcn: dropout, filters=hid_dim
    parser.add_argument('--kernel_size', type=int, default=4)
    ## sand: num_layers, hid_dim, num_heads, dropout
    parser.add_argument('--r', type=int, default=24)
    parser.add_argument('--M', type=int, default=12)
    ## grud: hid_dim, dropout
    parser.add_argument('--max_timesteps', type=int, default=880)
    # parser.add_argument('--max_timesteps', type=int, default=8654)
    ## interpnet: hid_dim
    parser.add_argument('--hours_look_ahead', type=int, default=24)
    parser.add_argument('--ref_points', type=int, default=24)

    # training/eval realated arguments
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=2196)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--print_train_loss_every', type=int, default=100)
    parser.add_argument('--validate_after', type=int, default=-1)
    parser.add_argument('--validate_every', type=int, default=None)

    args = parser.parse_args()
    return args


def set_output_dir(args: argparse.Namespace) -> None:
    """Function to automatically set output dir 
    if it is not passed in args."""
    if args.output_dir is None:
        if args.pretrain:
            args.output_dir = './outputs/' + args.dataset + '/' + args.output_dir_prefix + 'pretrain/'
        else:
            if args.load_ckpt_path is not None:
                args.output_dir_prefix = 'finetune_' + args.output_dir_prefix
            args.output_dir = './outputs/' + args.dataset + '/' + args.output_dir_prefix
            args.output_dir += args.model_type
            if args.model_type == 'strats':
                for param in ['num_layers', 'hid_dim', 'num_heads', 'dropout', 'attention_dropout', 'lr']:
                    args.output_dir += ',' + param + ':' + str(getattr(args, param))
            # for param in ['train_frac', 'run']:
            #     args.output_dir += '|' + param + ':' + str(getattr(args, param))
    os.makedirs(args.output_dir, exist_ok=True)
    # os.makedirs("./outputs/run", exist_ok=True)


if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n' + str(args))
    args.device = torch.device('cuda')
    set_all_seeds(args.seed + int(args.run.split('o')[0]))
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin')

    # load data
    dataset = PretrainDataset(args) if args.pretrain == 1 else Dataset(args)

    # load model
    model_class = {'strats': Strats, 'istrats': Strats, 'gru': GRU_TS, 'tcn': TCN_TS,
                   'sand': SAND, 'grud': GRUD_TS, 'interpnet': InterpNet, 'ehrmamba': EHR_Mamba, 'duett': DuETT, 'triplet_mamba': Triplet_Mamba}
    model = model_class[args.model_type](args)
    model.to(args.device)
    count_parameters(args.logger, model)
    if args.load_ckpt_path is not None:
        curr_state_dict = model.state_dict()
        pt_state_dict = torch.load(args.load_ckpt_path)
        for k, v in pt_state_dict.items():
            if k in curr_state_dict:
                curr_state_dict[k] = v
        model.load_state_dict(curr_state_dict)

    # training loop
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = num_train / args.train_batch_size
    args.logger.write('\nNo. of training batches per epoch = '
                      + str(num_batches_per_epoch))
    args.max_steps = int(round(num_batches_per_epoch) * args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(np.ceil(num_batches_per_epoch))
    cum_train_loss, num_steps, num_batches_trained = 0, 0, 0
    wait, patience_reached = args.patience, False
    best_val_metric = -np.inf
    best_val_res, best_test_res = None, None
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    train_bar = tqdm(range(args.max_steps))
    evaluator = PretrainEvaluator(args) if args.pretrain==1 else Evaluator(args)
    model_loss = []
    model_perf = []

    # results before any training
    if args.validate_after < 0:
        results = evaluator.evaluate(model, dataset, 'val', train_step=-1)
        if not (args.pretrain):
            evaluator.evaluate(model, dataset, 'eval_train', train_step=-1)
            evaluator.evaluate(model, dataset, 'test', train_step=-1)

    model.train()
    for step in train_bar:
        # load batch
        batch = dataset.get_batch()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # forward pass
        loss = model(**batch)

        # backward pass
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # add to cum loss
        cum_train_loss += loss.item()
        num_steps += 1
        num_batches_trained += 1

        # Log training losses.
        train_bar.set_description(str(np.round(cum_train_loss / num_batches_trained, 5)))
        if (num_steps) % args.print_train_loss_every == 0:
            args.logger.write('\nTrain-loss at step ' + str(num_steps) + ': '
                              + str(cum_train_loss / num_batches_trained))
            cum_train_loss, num_batches_trained = 0, 0

        # run validatation
        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            # get metrics on test and validation splits
            val_res = evaluator.evaluate(model, dataset, 'val', train_step=step)
            if not (args.pretrain):
                evaluator.evaluate(model, dataset, 'eval_train', train_step=step)
                test_res = evaluator.evaluate(model, dataset, 'test', train_step=step)
                model_loss.append( test_res['train_loss'] )
                model_perf.append( test_res['auroc'] )
            else:
                test_res = None
                model_loss.append( -1 * val_res['loss_neg'] )
            model.train(True)

            # Save ckpt if there is an improvement.
            curr_val_metric = val_res['loss_neg'] if args.pretrain \
                else val_res['auprc'] + val_res['auroc']
            if curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                best_val_res, best_test_res = val_res, test_res
                args.logger.write('\nSaving ckpt at ' + model_path_best)
                torch.save(model.state_dict(), model_path_best)
                wait = args.patience
            else:
                wait -= 1
                args.logger.write('Updating wait to ' + str(wait))
                if wait == 0:
                    args.logger.write('Patience reached')
                    break
    if not args.pretrain:
        embeddings = evaluator.get_embeddings( model, dataset, 'test' )
        os.makedirs(args.output_dir, exist_ok=True)
        outfile = os.path.join( args.output_dir, f'embeddings.pt' )
        torch.save(embeddings, outfile )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot( model_perf )
        ax2.plot( model_loss )
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.plot( model_loss )
        plt.show()

    # print final res
    args.logger.write('Final val res: ' + str(best_val_res))
    if best_test_res:
        args.logger.write('Final test res: ' + str(best_test_res))
