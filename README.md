

# AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning

This project is a fork of [facebookresearch/AdaTT](https://github.com/facebookresearch/AdaTT), with additional adaptations for training and inference deployment, as well as extra feature engineering for industrial scenarios.


AdaTT is a PyTorch-based library for multitask learning, focusing on recommendation and classification tasks. It implements the AdaTT architecture and several popular multitask learning baselines, providing a flexible and extensible framework for research and production.

## Quick Start

### Training

```bash
python train.py --data_path <TRAIN_DATA_PATH> --batch_size 128 --epochs 15 --use_focal_loss --is_flatten --model_tag <MODEL_TAG>
```

### Inference

```bash
python predict.py --pred_data_path <PRED_DATA_PATH> --model_key <MODEL_KEY> --data_tag <DATA_TAG>
```

## Configuration setting

### train.py Arguments

- `data_path` (str): Path to the training dataset file (parquet format).
- `label_cols` (list of str): Target label columns, e.g. `['label_register', 'label_apply', 'label_credit']`.
- `sample_ratios` (list of float): Sampling ratio for each task, default `[0.3, 0.5, 1, 1]`.
- `varlen_sparse_features` (list of str or None): Variable-length sparse feature names. If not set, default features will be used.
- `batch_size` (int): Batch size for training, default `128`.
- `use_focal_loss` (bool): Whether to use focal loss. If true, `pos_weights` will be ignored.
- `max_varlen_len` (int): Maximum length for variable-length features, default `50`.
- `dense_norm` (str): Normalization method for dense features, options: `'batch'` or `'layer'`, default `'batch'`.
- `emb_dim` (int): Embedding dimension for sparse features, default `6`.
- `num_task_experts` (int): Number of task-specific experts, default `2`.
- `expert_out_dims` (list of list of int): Output dimensions for each expert layer, e.g. `[[64, 64]]`.
- `num_shared_experts` (int): Number of shared experts, default `1`.
- `task_mlp` (list of int): Hidden layer sizes for task-specific MLP, default `[32, 32]`.
- `activation_type` (str): Activation function type, default `'DICE'`.
- `epochs` (int): Number of training epochs, default `10`.
- `pos_weights` (list of float or None): Positive sample weights for each label column. Length must match `label_cols`. Default is `None`.
- `is_flatten` (bool): Whether to flatten specific variable-length sparse features.
- `if_sample` (bool): Whether to perform sampling for high-frequency samples.

> For more details and usage, see `train.py` or run `python train.py --help`.


## Features

- **State-of-the-art Multitask Models**: Includes AdaTT, MMoE, Multi-level MMoE, PLE, Cross-stitch, and Shared-bottom architectures.
- **Flexible Data Pipeline**: Supports dense, sparse, and variable-length features, with efficient sampling and preprocessing utilities.
- **Custom Loss & Metrics**: Built-in support for Focal Loss, sample weighting, and common metrics (AUC, recall, precision, KS-score).
- **Easy Training & Inference**: Simple scripts for training and prediction with configurable arguments.
- **Extensible Design**: Modular codebase for easy extension and integration into larger systems.


## Supported Models

- **AdaTT** ([KDD'23 Paper](https://doi.org/10.1145/3580305.3599769))
- **MMoE** ([Paper](https://dl.acm.org/doi/10.1145/3219819.3220007))
- **Multi-level MMoE** (extension)
- **PLE** ([Paper](https://doi.org/10.1145/3383313.3412236))
- **Cross-stitch** ([Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf))
- **Shared-bottom** ([Paper](https://link.springer.com/article/10.1023/a:1007379606734))

All models can be selected and configured via the `CentralTaskArch` class.

## Citation

If you use AdaTT or find it helpful, please cite:

```bibtex
@article{li2023adatt,
  title={AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations},
  author={Li, Danwei and Zhang, Zhengyu and Yuan, Siyang and Gao, Mingze and Zhang, Weilin and Yang, Chaofei and Liu, Xi and Yang, Jiyan},
  journal={arXiv preprint arXiv:2304.04959},
  year={2023}
}
```

## License

This project is licensed under the MIT License.

## References

- [AdaTT Paper (KDD'23)](https://doi.org/10.1145/3580305.3599769)
- [arXiv Preprint](https://arxiv.org/abs/2304.04959)
- [Project Slides](https://drive.google.com/file/d/1I8XpxPxwhP9KXuztEguYkuMM10kiJDS7/view?usp=sharing)