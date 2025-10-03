import os
import glob
import torch
import argparse
import numpy as np
import pandas as pd

from loguru import logger
from dataset import MultiTaskDataset
from torch.utils.data import DataLoader

from tools import hash_mapping, parse_varlen
from model import MtlConfigs, ArchInputs, CentralTaskArch


def main(pred_data_path,
         model_key,
         data_tag,
         target_cols=['label_register', 'label_apply', 'label_credit', 'label_good'],
         fix_vocab_size=10000,
         varlen_vocab_size=10000,
         max_varlen_len=10,
         batch_size=128,
         emb_dim=6,
         num_task_experts=2,
         num_shared_experts=1,
         expert_out_dims=[[64, 64]],
         task_mlp=[32, 32],
         activation_type='DICE',
         dense_norm='batch',
         varlen_sparse_features=None,
         is_flatten=False,
         ):
    os.makedirs('results', exist_ok=True)
    logger.add(f"logs/{model_key}/predict.log", rotation="10 MB", encoding="utf-8")
    logger.info("Loading inference data...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    parquet_files = sorted(glob.glob(os.path.join(pred_data_path, '*.parquet')))
    assert len(parquet_files) > 0, f"文件夹 data 下未找到parquet文件"

    model_path = f"model/{model_key}/best.pt"

    # read sample df to infer feature names
    sample_df = pd.read_parquet(parquet_files[0])
    sample_df = sample_df.head(1000)

    fix_sparse_features = [
        {"name": col, "embedding_name": col}
        for col in sample_df.columns
    ]

    if varlen_sparse_features is None:
        varlen_sparse_features = [
        ]

    dense_features = [col for col in sample_df.columns if col not in varlen_sparse_features and col not in [f["name"] for f in fix_sparse_features]]

    dense_dim = len(dense_features)
    fix_sparse_dim = len(fix_sparse_features)
    varlen_sparse_dim = len(varlen_sparse_features)
    num_tasks = len(target_cols)

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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    result_path = f"results/{data_tag}.csv"
    first_chunk = True

    batch_size = 40
    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i+batch_size]
        batch_chunks = [pd.read_parquet(f) for f in batch_files]
        chunk = pd.concat(batch_chunks, ignore_index=True)

        # fix_sparse提前hash
        for feat in fix_sparse_features:
            col = feat["name"]
            chunk[col] = chunk[col].map(lambda x: hash_mapping(x, fix_vocab_size) if pd.notna(x) else 0)

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
            arrs = chunk[col].map(
                split_and_flatten if is_flatten else lambda x: list(parse_varlen(x))
            )
            # 2. 统一空值为空列表
            arrs = arrs.apply(lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) > 0 else [])
            # 3. 哈希化
            arrs = arrs.apply(lambda arr: hash_arr(arr, varlen_vocab_size))
            # 4. pad/trunc
            chunk[col] = arrs.apply(lambda arr: pad_trunc(arr, max_varlen_len))

        # 转 tensor
        dense_tensor = torch.tensor(chunk[dense_features].values, dtype=torch.float32)
        fix_sparse_tensor = torch.tensor(
            np.stack([chunk[feat["name"]].values for feat in fix_sparse_features], axis=1), dtype=torch.int64)
        varlen_tensor = torch.tensor(
            np.stack([np.stack(chunk[feat["name"]]) for feat in varlen_sparse_features], axis=1), dtype=torch.int64)

        predict_dataset = MultiTaskDataset(dense_tensor, fix_sparse_tensor, varlen_tensor)
        predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds = [[] for _ in range(num_tasks)]
        with torch.no_grad():
            for dense, fix_sparse, varlen_sparse in predict_loader:
                dense = dense.to(device)
                fix_sparse = fix_sparse.to(device)
                varlen_sparse = varlen_sparse.to(device)
                outputs = model(dense, fix_sparse, varlen_sparse)
                for i in range(num_tasks):
                    probs = torch.sigmoid(outputs[i]).cpu().numpy()
                    all_preds[i].append(probs)

        all_preds = [np.concatenate(preds, axis=0) for preds in all_preds]

        result_df = pd.DataFrame({'id': chunk['id'].values})
        for i, col in enumerate(target_cols):
            result_df[f"{col.replace('label_', '')}_score"] = all_preds[i].reshape(-1)
            
        result_df.to_csv(result_path, mode='a', header=first_chunk, index=False)
        first_chunk = False 

        # logger.info(f"Chunk processed and predictions made, result shape: {result_df.shape}")
    logger.info(f"Prediction results saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict with trained model')
    parser.add_argument('--pred_data_path', type=str, required=True, help='预测数据文件夹路径（parquet文件夹）')
    parser.add_argument('--model_key', type=str, required=True, help='推理数据集路径')
    parser.add_argument('--data_tag', type=str, required=True, help='预测model tag')
    parser.add_argument('--is_flatten', action='store_true', help='是否对varlen特征进行flatten处理')
    parser.add_argument('--target_cols', type=str, nargs='+', default=['label_register', 'label_apply', 'label_credit', 'label_good'])
    parser.add_argument('--fix_vocab_size', type=int, default=10000)
    parser.add_argument('--varlen_vocab_size', type=int, default=10000)
    parser.add_argument('--max_varlen_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=6)
    parser.add_argument('--num_task_experts', type=int, default=2)
    parser.add_argument('--num_shared_experts', type=int, default=1)
    parser.add_argument('--expert_out_dims', type=str, nargs='+', default=['64,64'])
    parser.add_argument('--task_mlp', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--activation_type', type=str, default='DICE')
    parser.add_argument('--dense_norm', type=str, default='batch')
    parser.add_argument('--varlen_sparse_features', type=str, nargs='+', default=None)
    args = parser.parse_args()

    # expert_out_dims 处理
    args.expert_out_dims = [[int(x) for x in group.split(',')] for group in args.expert_out_dims]
    

    pred_data_path = args.pred_data_path
    model_key = args.model_key
    data_tag = args.data_tag
    target_cols = args.target_cols
    fix_vocab_size = args.fix_vocab_size
    varlen_vocab_size = args.varlen_vocab_size
    max_varlen_len = args.max_varlen_len
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    num_task_experts = args.num_task_experts
    num_shared_experts = args.num_shared_experts
    expert_out_dims = args.expert_out_dims
    task_mlp = args.task_mlp
    activation_type = args.activation_type
    dense_norm = args.dense_norm
    varlen_sparse_features = args.varlen_sparse_features
    is_flatten = args.is_flatten

    main(
         pred_data_path=pred_data_path,
         model_key=model_key,
         data_tag=data_tag,
         target_cols=target_cols,
         fix_vocab_size=fix_vocab_size,
         varlen_vocab_size=varlen_vocab_size,
         max_varlen_len=max_varlen_len,
         batch_size=batch_size,
         emb_dim=emb_dim,
         num_task_experts=num_task_experts,
         num_shared_experts=num_shared_experts,
         expert_out_dims=expert_out_dims,
         task_mlp=task_mlp,
         activation_type=activation_type,
         dense_norm=dense_norm,
         varlen_sparse_features=varlen_sparse_features,
         is_flatten=is_flatten)
    
    # python predict.py --pred_data_path /mnt/home/yangzhou23/data/Exper-27-PRED-DATA-20250925 --model_key Exper-27-ADATT-HFQ-SEED-25M5-M9-20250925 --data_tag TMP_Exper-27-ADATT-HFQ-SEED-25M5-M9_20250927 
