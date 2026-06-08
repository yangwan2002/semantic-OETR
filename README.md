# semantic-OETR

基于 [OETR](https://github.com/AttentiveSon/OETR)（AAAI 2022）扩展的空天地 relay **重叠区域估计**模型，面向 CARLA-Air 实验：在原始 bbox 回归基础上，增加稠密重叠 mask 预测、可选 DINOv2 语义融合，以及缓解 UAV/UGV 与 relay 尺度差异的 Position Encoding（PE）。

原始 OETR 使用 Transformer 估计图像对的重叠框；本仓库在其上增加了 mask head、soft+planar GT 监督，以及 UAV/UGV relay 微调配置，用于论文中的空天地特征匹配与融合流程。

<p align="center">
  <img src="doc/network.png" width="60%"/>
</p>

## 本仓库新增内容

| 模块 | 说明 |
|------|------|
| **Mask head** | 在 bbox 回归之外，输出稠密重叠 mask（BCE + Dice） |
| **SemDINO 融合** | 可选 `dinov2_vitb14` 特征，配合 `mask_align` 语义损失 |
| **Scale PE** | 可学习位置编码，缓解 UAV–relay / UGV–relay 尺度不一致 |
| **CARLA 数据加载** | `carla_fullimage_pairs`：全图 relay 图像对 + mask |
| **Soft+planar GT** | 在 soft overlap + planar depth GT 上微调（UAV / UGV 两支） |
| **推理脚本** | `infer_pair.py`：任意图像对的 bbox + mask 导出 |

## 目录结构

```
configs/baseline/     # OETR 与 CARLA 训练配置
dloc/                 # 与定位 / 匹配流水线集成的 overlap 模块
infer_pair.py         # 独立图像对推理入口
scripts/              # UAV / UGV soft+planar PE 训练脚本
src/                  # 模型、损失、数据集、验证逻辑
weights/              # 权重放这里（不纳入 git）
```

## 环境依赖

训练（建议 GPU 服务器）：

```bash
pip install -r requirements.txt
```

仅做轻量图像对推理：

```bash
pip install -r requirements-infer.txt
```

实测环境为 Python 3.10+、PyTorch 2.x。原版 OETR 文档写的是 Python 3.6；本 fork 在训练服务器上使用较新的 torch / timm / kornia 组合。

## CARLA-Air 训练

数据通常导出为序列目录下的 JSONL 配对列表，例如：

```
<sequence_dir>/oetr_fullimage_uavrelay_soft_planar_clean_f005/pairs_train.jsonl
<sequence_dir>/oetr_fullimage_uavrelay_soft_planar_clean_f005/pairs_val.jsonl
```

训练前请在配置文件中修改 `DATASET.DATA_ROOT` 与 `LIST_PATH`。

### 推荐配置链路

1. **Mask 基线** — `configs/baseline/carla_fullimg_uavrelay_1280_clean_f005_maskhead.py`
2. **SemDINO mask-align（最佳）** — `carla_fullimg_uavrelay_1280_clean_f005_maskhead_semdino_maskalign_v2_best.py`
3. **+ Scale PE** — `carla_fullimg_uavrelay_1280_clean_f005_maskhead_semdino_maskalign_v2_best_pe_v1.py`
4. **UAV soft+planar 微调** — `carla_fullimg_uavrelay_soft_planar_finetune.py`
5. **UGV soft+planar 微调** — `carla_fullimg_ugvrelay_soft_planar_finetune.py`

单卡训练示例：

```bash
cd /path/to/semantic-OETR
python train.py configs/baseline/carla_fullimg_uavrelay_soft_planar_finetune.py
```

也可直接使用脚本：

```bash
bash scripts/train_uav_soft_planar_pe.sh
bash scripts/train_ugv_soft_planar_pe.sh
```

权重默认保存在 `OUTPUT/OETR/checkpoints/<cfg.OUTPUT>/`。

## 图像对推理

```bash
python infer_pair.py \
  --image1 /path/to/uav.png \
  --image2 /path/to/relay.png \
  --checkpoint weights/carla_uavrelay_soft_planar_pe_ep11/model_epoch_11.pth \
  --output-dir outputs/pair_demo
```

默认配置：`configs/baseline/carla_fullimg_uavrelay_soft_planar_finetune_pe_infer.py`。

## 原版 OETR（MegaDepth）

原版 MegaDepth 训练与验证流程仍保留，见 `configs/baseline/oetr_config.py`。

### MegaDepth 数据

- [MegaDepth depth maps](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz)
- [D2-Net 预处理图像](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB)
- 配对列表：[Google Drive](https://drive.google.com/drive/folders/1xN56olSJIfqZ4i35ENoNeyt8Wi2m7iRA?usp=sharing)

```bash
ln -s /path/to/megadepth/* ./dataset/megadepth
```

### 原版训练

```bash
scripts/train.sh
```

## 定位推理流水线（hloc）

见 [dloc/README.md](dloc/README.md) 与 [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)。

## 引用

若使用原始 OETR 方法，请引用：

```bibtex
@inproceedings{chen2022guide,
  title={Guide Local Feature Matching by Overlap Estimation},
  author={Chen, Ying and Huang, Dihe and Xu, Shang and Liu, Jianlin and Liu, Yong},
  booktitle={AAAI},
  year={2022}
}
```

## 许可证

见 [license.txt](license.txt)。本仓库在原始 OETR 代码结构之上扩展，用于 CARLA-Air 相关研究。
