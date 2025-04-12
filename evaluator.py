import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from tqdm import tqdm
from torcheval.metrics import BinaryAccuracy


class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step):
        # self.args.logger.write('\nEvaluating on split = ' + split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            del batch['labels']
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch)[0].cpu())
        true, pred = torch.cat(true), torch.cat(pred)
        metric = BinaryAccuracy()
        precision, recall, thresholds = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        metric.update( pred, true )
        acc = metric.compute()
        # acc = accuracy_score( true, pred )
        result = {'auroc': np.round( roc_auc, 4 ), 'auprc': np.round( pr_auc, 4 ), 'minrp': np.round( minrp, 4 ), 'acc': np.round( acc.float(), 4 )}
        # if train_step is not None:
        #     self.args.logger.write('Result on ' + split + ' split at train step '
        #                            + str(f'{train_step:.4f}') + ': ' + str(result))
        return result

    def get_embeddings( self, model, dataset, split ):
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()
        
        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size),
                    desc='running forward pass')
        # embs = torch.empty((0,8), dtype=torch.float64)
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            # true.append(batch['labels'])
            del batch['labels']
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                e = model(**batch)[1].cpu()
            if start:
                embs = e
            else:
                embs = torch.cat( ( embs, e ), 0 ) 
        return embs
