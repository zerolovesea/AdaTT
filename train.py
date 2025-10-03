import os
import gc
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from loguru import logger
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from dataset import MultiTaskDataset
from model import MtlConfigs, ArchInputs, CentralTaskArch, FocalLoss
from tools import hash_mapping, parse_varlen, compute_sample_weights, ks_score, sample_pos_neg

def train(data_path='',
          label_cols=[],
          if_sample=True,
          sample_ratios=[],
          varlen_sparse_features=[],
          batch_size=128,
          use_focal_loss=False,
          max_varlen_len=10,
          dense_norm="batch",
          emb_dim=6,
          num_shared_experts=1,
          num_task_experts=2,
          expert_out_dims=[[64, 64]],
          task_mlp=[32, 32],
          activation_type="DICE",
          epochs=10,
          pos_weights=None,
          is_flatten=False,
          ):
    
    logger.info('Starting data processing and model training...')
    logger.info(f'Data path: {data_path}')
    logger.info(f'Label columns: {label_cols}')

    # ------------------- 读取数据 -------------------
    data = pd.read_parquet(data_path)
    logger.info(f"Data shape: {data.shape}")

    assert len(label_cols) == len(sample_ratios), "Length of label_cols must match length of sample_ratios"

    if if_sample:
        # avoid too much sample in early stage tasks
        logger.info(f'Sampling')
        sampled_data_list = []
        for col, ratio in zip(label_cols, sample_ratios):
            sampled = sample_pos_neg(data, col, ratio)
            sampled_data_list.append(sampled)
        target_cols = label_cols
        del data
        gc.collect()
        logger.info('Merging sampled data')
        data = pd.concat(sampled_data_list).drop_duplicates(subset=['phone']).reset_index(drop=True)
    else:
        target_cols = label_cols

    if pos_weights is None:
        pos_weights = [1.0] * len(label_cols)

    assert len(pos_weights) == len(label_cols), "pos_weights长度需与label_cols一致"
    sample_weights = compute_sample_weights(data, target_cols)
    logger.info(f"After sampling Data shape: {data.shape}")

    logger.info('Target distribution:')
    for col in target_cols:
        logger.info(f"  {col}: {data[col].value_counts().to_dict()}")

    # fix sparse features are not sharing embedding lookup table
    fix_sparse_features = [
        {"name": col, "embedding_name": col}  
        for col in data.columns if 'tag' in col
    ]
    if varlen_sparse_features is None:
        varlen_sparse_features = [
        ]

    dense_features = [col for col in data.columns if col not in varlen_sparse_features and col not in [f["name"] for f in fix_sparse_features]]

    fix_vocab_size = 10000
    varlen_vocab_size = 10000

    num_tasks = len(target_cols)
    logger.info(f"Number of tasks: {num_tasks}, target columns: {target_cols}")
    logger.info(f"Number of features - Dense: {len(dense_features)}, Fix sparse: {len(fix_sparse_features)}, Varlen sparse: {len(varlen_sparse_features)}")

    logger.info('Hashing fix sparse features...')

    for feat in fix_sparse_features:
        col = feat["name"]
        data[col] = data[col].apply(lambda x: hash_mapping(x, fix_vocab_size) if pd.notna(x) else 0)

    for feat in varlen_sparse_features:
        col = feat["name"]
        logger.info(f"{'Flattening and ' if is_flatten else ''}Hashing varlen sparse feature: {col}")

        def split_and_flatten(x):
            arr = list(parse_varlen(x))
            result = []
            for item in arr:
                if isinstance(item, str):
                    result.extend(item.split('-'))
                else:
                    result.append(item)
            return [v for v in result if v != '']

        def hash_arr(arr, vocab_size):
            return np.array([hash_mapping(v, vocab_size) for v in arr], dtype=np.int32)

        def pad_trunc(arr, max_len):
            arr = np.array(arr, dtype=np.int32)
            if len(arr) >= max_len:
                return arr[:max_len].tolist()
            return np.pad(arr, (0, max_len - len(arr)), 'constant').tolist()
        
        # 1. parse & flatten
        arrs = data[col].map(
            split_and_flatten if is_flatten else lambda x: list(parse_varlen(x))
        )
        # 2. 统一空值为空列表
        arrs = arrs.apply(lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) > 0 else [])
        # 3. 哈希化
        arrs = arrs.apply(lambda arr: hash_arr(arr, varlen_vocab_size))
        # 4. pad/trunc
        data[col] = arrs.apply(lambda arr: pad_trunc(arr, max_varlen_len))

    logger.info('Convert pandas to tensor...')
    dense_tensor = torch.tensor(data[dense_features].values, dtype=torch.float32)
    fix_sparse_tensor = torch.tensor(
        np.stack([data[feat["name"]].values for feat in fix_sparse_features], axis=1), dtype=torch.int64)
    varlen_tensor = torch.tensor(
        np.stack([np.stack(data[feat["name"]]) for feat in varlen_sparse_features], axis=1), dtype=torch.int64)
    targets_tensor = torch.tensor(data[target_cols].values, dtype=torch.float32)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    dataset = MultiTaskDataset(
        dense_tensor, fix_sparse_tensor, varlen_tensor, targets_tensor, sample_weights_tensor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dense_dim = len(dense_features)
    fix_sparse_dim = len(fix_sparse_features)
    varlen_sparse_dim = len(varlen_sparse_features)

    mtl_configs = MtlConfigs(
        mtl_model="att_sp",
        num_task_experts=num_task_experts,
        num_shared_experts=num_shared_experts,
        expert_out_dims=expert_out_dims,
        self_exp_res_connect=True,
    )
    opts = ArchInputs(
        num_task=num_tasks,
        task_mlp=task_mlp,
        mtl_configs=mtl_configs,
        activation_type=activation_type,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = CentralTaskArch(
        mtl_configs,
        opts,
        dense_dim=dense_dim,
        fix_sparse_dim=fix_sparse_dim,
        varlen_sparse_dim=varlen_sparse_dim,
        fix_vocab_size=fix_vocab_size,
        varlen_vocab_size=varlen_vocab_size,
        varlen_len=max_varlen_len,
        emb_dim=emb_dim,
        dense_norm=dense_norm,
        fix_sparse_features=fix_sparse_features,
        varlen_sparse_features=varlen_sparse_features,
    )
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    logger.info(f"Starting training, total epochs: {epochs}")

    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        auc_storage = [[[], []] for _ in range(num_tasks)]

        for dense, fix_sparse, varlen_sparse, targets, batch_weights in dataloader:
            dense = dense.to(device, non_blocking=True)
            fix_sparse = fix_sparse.to(device, non_blocking=True)
            varlen_sparse = varlen_sparse.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(dense, fix_sparse, varlen_sparse)
            losses = []

            # 正样本加权
            for i in range(num_tasks):
                out = outputs[i].view(-1, 1)
                tgt = targets[:, i].view(-1, 1)
                if use_focal_loss:
                    loss_vec = focal_loss_fn(out, tgt)
                    weighted_loss = loss_vec.view(-1).mean()  # focal loss handles weighting
                else:
                    loss_vec = nn.BCEWithLogitsLoss(reduction='none')(out, tgt)
                    # 正样本加权
                    pos_mask = (tgt.view(-1) == 1)
                    batch_weights[:, i][pos_mask] = batch_weights[:, i][pos_mask] * pos_weights[i] # sample weight times pos weight
                    weighted_loss = (loss_vec.view(-1) * batch_weights[:, i]).mean()
                losses.append(weighted_loss)

                out_np, tgt_np = out.detach().cpu().numpy().ravel(), tgt.detach().cpu().numpy().ravel()
                auc_storage[i][0].extend(tgt_np)
                auc_storage[i][1].extend(out_np)

            loss = torch.stack(losses).sum()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        # compute metrics
        auc_vals, ks_vals, recall_vals, precision_vals = [], [], [], []
        all_y_true, all_y_pred = [], []

        for i in range(num_tasks):
            y_true = np.array(auc_storage[i][0])
            y_pred = np.array(auc_storage[i][1])
            y_pred_prob = 1 / (1 + np.exp(-y_pred))  # sigmoid

            # AUC
            if len(np.unique(y_true)) < 2:
                auc_vals.append(np.nan)
                ks_vals.append(np.nan)
                recall_vals.append(np.nan)
                precision_vals.append(np.nan)
            else:
                auc_vals.append(roc_auc_score(y_true, y_pred))
                ks_vals.append(ks_score(y_true, y_pred_prob))
                recall_vals.append(recall_score(y_true, y_pred_prob > 0.5))
                precision_vals.append(precision_score(y_true, y_pred_prob > 0.5))
            all_y_true.append(y_true.reshape(-1, 1))
            all_y_pred.append((y_pred_prob > 0.5).reshape(-1, 1))

        all_y_true = np.concatenate(all_y_true, axis=1)
        all_y_pred = np.concatenate(all_y_pred, axis=1)

        all_true = (all_y_true == 1).all(axis=1)
        all_pred = (all_y_pred == 1).all(axis=1)
        overall_recall = recall_score(all_true, all_pred)
        overall_precision = precision_score(all_true, all_pred)

        metric_strs = []
        for i in range(num_tasks):
            metric_strs.append(
                f"Task{i+1} ({target_cols[i]}):\n"
                f"  AUC   : {(auc_vals[i] if not np.isnan(auc_vals[i]) else 0):.4f}\n"
                f"  KS    : {(ks_vals[i] if not np.isnan(ks_vals[i]) else 0):.4f}\n"
                f"  Recall: {(recall_vals[i] if not np.isnan(recall_vals[i]) else 0):.4f}\n"
                f"  Prec  : {(precision_vals[i] if not np.isnan(precision_vals[i]) else 0):.4f}"
            )
        metric_strs.append(f"OverallRecall: {overall_recall:.4f}\nOverallPrec  : {overall_precision:.4f}")
        logger.info(
            f"Epoch {epoch+1}/{epochs}, TotalLoss: {np.mean(epoch_losses):.4f}\n" +
            '\n'.join(metric_strs)
        )

        torch.save(model.state_dict(), f"{model_dir}/checkpoint.pt")

        if epoch == 0:
            best_overall_prec = overall_precision
            torch.save(model.state_dict(), f"{model_dir}/best.pt")
        else:
            if overall_precision > best_overall_prec:
                best_overall_prec = overall_precision
                torch.save(model.state_dict(), f"{model_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predicting with trained models')
    parser.add_argument('--model_tag', type=str, help='model path')
    parser.add_argument('--data_path', type=str, default=None, help='训练数据路径')
    parser.add_argument('--label_cols', type=str, nargs='+', default=['label_register', 'label_apply', 'label_credit', 'label_good'], help='目标标签列名列表')
    parser.add_argument('--if_sample', action='store_true', help='是否进行正负样本采样')
    parser.add_argument('--sample_ratios', type=float, nargs='+', default=[0.3, 0.5, 1, 1], help='每个标签的采样比例列表')
    parser.add_argument('--varlen_sparse_features', type=str, nargs='+', default=None, help='可变长度稀疏特征名称列表')
    parser.add_argument('--batch_size', type=int, default=128, help='训练的批次大小')
    parser.add_argument('--is_flatten', action='store_true', help='是否对varlen_sparse_features进行拆分展平')

    parser.add_argument('--use_focal_loss', action='store_true', help='是否使用focal loss')
    parser.add_argument('--max_varlen_len', type=int, default=50, help='可变长度特征的最大长度')

    parser.add_argument('--dense_norm', type=str, default='batch', help='稠密特征的归一化方法')
    parser.add_argument('--emb_dim', type=int, default=6, help='稀疏特征的嵌入维度')
    parser.add_argument('--num_task_experts', type=int, default=2, help='任务特定专家的数量')
    parser.add_argument('--num_shared_experts', type=int, default=1, help='共享专家的数量')
    parser.add_argument('--expert_out_dims', type=str, nargs='+', default=['64,64'], help='每个专家层的输出维度，如 64,64 32,32') # 会被包装成二维列表[[64,64]]
    parser.add_argument('--task_mlp', type=int, nargs='+', default=[32, 32], help='任务特定MLP的隐藏层大小')
    parser.add_argument('--activation_type', type=str, default='DICE', help='激活函数类型')
    parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
    parser.add_argument('--pos_weights', type=float, nargs='+', default=None, help='正样本权重列表，长度需与label_cols一致')
    args = parser.parse_args()

    model_tag = args.model_tag
    data_path = args.data_path
    label_cols = args.label_cols
    sample_ratios = args.sample_ratios
    varlen_sparse_features = args.varlen_sparse_features
    batch_size = args.batch_size
    use_focal_loss = args.use_focal_loss
    max_varlen_len = args.max_varlen_len
    dense_norm = args.dense_norm
    emb_dim = args.emb_dim
    num_task_experts = args.num_task_experts
    num_shared_experts = args.num_shared_experts
    pos_weights = args.pos_weights
    is_flatten = args.is_flatten
    if_sample = args.if_sample

    expert_out_dims = []
    for group in args.expert_out_dims:
        expert_out_dims.append([int(x) for x in group.split(',')])

    task_mlp = args.task_mlp
    activation_type = args.activation_type
    epochs = args.epochs

    model_dir = f"model/{model_tag}"
    os.makedirs(model_dir, exist_ok=True)
    log_dir = f"logs/{model_tag}"
    os.makedirs(log_dir, exist_ok=True)

    logger.add(f"{log_dir}/train.log", rotation="10 MB", encoding="utf-8")

    train(
        data_path=data_path,
        label_cols=label_cols,
        sample_ratios=sample_ratios,
        varlen_sparse_features=varlen_sparse_features,
        batch_size=batch_size,
        use_focal_loss=use_focal_loss,
        max_varlen_len=max_varlen_len,
        dense_norm=dense_norm,
        emb_dim=emb_dim,
        num_task_experts=num_task_experts,
        expert_out_dims=expert_out_dims,
        num_shared_experts=num_shared_experts,
        task_mlp=task_mlp,
        activation_type=activation_type,
        epochs=epochs,
        pos_weights=pos_weights,
        is_flatten=is_flatten,
        if_sample=if_sample,)

    # python train.py --data_path /dataset/test_dataset --batch_size 128 --epochs 15 --use_focal_loss --is_flatten --model_tag test_model