import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from tqdm import tqdm
from torcheval.metrics import BinaryAccuracy
import torch.nn.functional as F


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
                pred.append(model(**batch).cpu())
        true, pred = torch.cat(true), torch.cat(pred)
        bce_loss = F.binary_cross_entropy_with_logits( pred, true )
        metric = BinaryAccuracy()
        precision, recall, thresholds = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        metric.update( pred, true )
        acc = metric.compute()
        result = {'auroc': np.round( roc_auc, 4 ), 'auprc': np.round( pr_auc, 4 ), 'minrp': np.round( minrp, 4 ), 'acc': np.round( acc.float(), 4 ), 'train_loss': np.round( bce_loss.item(), 4 ) }
        return result

    def get_embeddings( self, model, dataset, split ):
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()
        
        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size),
                    desc='running forward pass')
        embs = []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            # true.append(batch['labels'])
            del batch['labels']
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                l = model(**batch).cpu()
                e = model.embedding
            embs.append( e ) 
        embs = torch.cat( embs, dim = 0 ) 
        return embs

    def get_var_att_map( self, model, dataset, split ):
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()
        
        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size), desc='running forward pass')
        id_to_name_map = {v: k for k, v in dataset.var_ind.items()}
        attention_by_name = {}
        
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            del batch['labels']
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch).cpu()
                attention_matrix_cpu = torch.squeeze(model.att_weights).detach().cpu()
                variable_ind_cpu = batch['varis']

                for b in range(variable_ind_cpu.shape[0]):
                    local_var_att_map = {}
                    seq_attention = attention_matrix_cpu[b]      # Get attention weights for the current sequence
                    seq_variable_ind = variable_ind_cpu[b]
                
                    unique_ids, inverse_indices = torch.unique(seq_variable_ind, return_inverse=True)
                    agg_att = torch.zeros(unique_ids.size(0), dtype=seq_attention.dtype, device=seq_attention.device)
                    agg_att.scatter_add_(0, inverse_indices.to(seq_attention.device), seq_attention)
                    
                    for i, var_id_tensor in enumerate(unique_ids):
                        var_id = var_id_tensor.item()
                        attn_sum = agg_att[i].item()

                        if abs(attn_sum) > torch.finfo(seq_attention.dtype).eps:
                            model.global_var_att_map[var_id] = model.global_var_att_map.get(var_id, 0.0) + attn_sum
                            local_var_att_map[id_to_name_map.get(var_id, f"Unknown_ID_{var_id}")] = attn_sum
                    model.local_var_att_map.append( local_var_att_map )

        for var_id, attn_weight in model.global_var_att_map.items():
            variable_name = id_to_name_map.get(var_id, f"Unknown_ID_{var_id}")
            attention_by_name[variable_name] = attn_weight
        return attention_by_name
