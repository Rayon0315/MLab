# MLab Train / Test Interface Guide

> 这份 README 只说明 **MLab 的训练与测试接口怎么用**。  
> 不解释网络结构、不记录实验结果、不讨论论文设计。
>
> 目标是让后续修改或新增网络时，只要先看这里，就能判断：
>
> - 应该调用哪个训练入口；
> - `--network` 应该怎么写；
> - 哪些数据目录必须传；
> - NAM / region mean / region loss 如何决定输入；
> - checkpoint / resume / test 如何对应；
> - 不同服务器应该使用哪套路径；
> - 什么情况下可以复用标准 `train.py / test.py`，什么情况下必须使用独立训练入口。

---

# 1. 总体约定

MLab 的实验接口尽量遵循下面的原则：

```text
Network
  ↓
train entry
  ↓
run-dir/
  ├── checkpoints/
  ├── logs/
  └── network_source/
  ↓
test entry
  ↓
predictions + metrics
```

绝大多数网络变体只改变：

```bash
--network models.networks.xxx
```

而 **不应该因为换了网络结构就复制一套 train/test**。

只有训练逻辑本身发生变化时，例如：

- 使用独立 loss；
- 使用弱监督训练协议；
- 使用特殊 pseudo-label / consistency 训练；
- trainer 行为不同；

才应该使用独立训练入口，例如：

```text
train.py
train_mae_loss.py
train_weak_xxx.py
...
```

但如果网络 forward 和 checkpoint 格式没有改变，测试一般仍然优先复用：

```bash
python test.py
```

---

# 2. 当前路径规则

这是最容易混淆的部分。

## 2.1 学校服务器

目前以下两条路线都按学校服务器规则处理：

- CFMNet backboned；
- 弱监督 VMamba。

工作目录：

```bash
/home/data2/MaLei/MLab
```

因此脚本开头通常写：

```bash
cd /home/data2/MaLei/MLab
```

数据集放在 MLab 内：

```text
/home/data2/MaLei/MLab/datasets/
```

脚本中通常使用相对路径：

```bash
datasets/EORSSD/...
datasets/ORSSD/...
datasets/ORSI-4199/...
```

---

## 2.2 AutoDL / 4090 上的 VMamba 路线

历史上标准 VMamba backboned 路线使用：

```bash
cd /home/MLab
```

数据主要位于：

```text
/root/autodl-tmp
```

因此这类脚本不要机械复制学校服务器的路径。

### 生成脚本时必须先确认

```text
这是：
1. 学校服务器 CFMNet？
2. 学校服务器弱监督 VMamba？
3. AutoDL 标准 VMamba？
```

如果属于前两项：

```bash
cd /home/data2/MaLei/MLab
```

如果属于第三项：

```bash
cd /home/MLab
```

数据路径再按对应环境填写。

---

# 3. Network 参数

MLab 通过 Python module path 选择网络。

例如：

```bash
--network models.networks.mambavision_small_progressive_region_direct_hier60_sod
```

对应：

```text
models/
└── networks/
    └── mambavision_small_progressive_region_direct_hier60_sod.py
```

不要写 `.py`：

```bash
# 正确
--network models.networks.xxx

# 错误
--network models/networks/xxx.py
```

训练和测试必须使用 **同一个 network module**：

```bash
train:
--network models.networks.xxx

test:
--network models.networks.xxx
```

否则 checkpoint 的参数结构可能无法匹配。

---

# 4. 标准训练入口：train.py

一般网络实验优先从：

```bash
python train.py
```

进入。

典型形式：

```bash
python train.py \
    --network models.networks.xxx \
    --run-dir runs/xxx \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --image-size 352 \
    --epochs 50 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
```

---

# 5. 标准训练参数

## 5.1 网络与输出目录

### `--network`

网络模块。

```bash
--network models.networks.xxx
```

### `--run-dir`

当前实验的唯一输出目录。

```bash
--run-dir runs/my_experiment
```

建议：

```text
runs/
└── my_experiment/
    ├── checkpoints/
    ├── logs/
    └── network_source/
```

不要让两个不同实验共用同一个 `run-dir`。

---

# 6. 训练数据接口

## 6.1 RGB 图像

```bash
--train-images datasets/EORSSD/train-images
```

---

## 6.2 GT mask

```bash
--train-masks datasets/EORSSD/train-labels
```

标准全监督 SOD 训练必须提供。

---

## 6.3 NAM 输入

当网络声明自己使用 NAM 时：

```bash
--train-nam datasets/EORSSD/...
```

具体目录按当前数据组织填写。

训练入口会根据模型的输入声明判断：

```text
model uses NAM
    ↓ yes
读取 --train-nam

model does not use NAM
    ↓
不把 NAM 送进 dataset/model
```

因此：

> 不要因为某个数据集目录里存在 NAM，就默认所有模型都要传 NAM。

反过来，如果网络实际需要 NAM，而脚本没有给正确目录，则训练无法正确运行。

---

## 6.4 Region Mean 输入

使用 NAMLab region mean 的网络需要：

```bash
--train-mean datasets/EORSSD/train-mean
```

例如：

```bash
python train.py \
    --network models.networks.mambavision_small_progressive_region_direct_hier60_sod \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-mean datasets/EORSSD/train-mean \
    ...
```

训练代码会读取模型声明的 mean hierarchy。

例如模型只要求：

```text
mean_60
```

就只应把对应 hierarchy 当作模型输入。

不要在 shell 里自行猜：

```text
mean_10
mean_20
mean_40
mean_60
...
```

模型需要哪些 hierarchy，以模型声明和 `engine/model_inputs.py` 的解析结果为准。

---

# 7. Region Mean 与 Region Loss 是两件事

这一点必须区分。

## 7.1 Region Mean 作为网络输入

例如：

```text
RGB
+
mean_60
↓
model forward
```

这时模型本身需要 `--train-mean`。

---

## 7.2 Region Mean 作为监督 target

当：

```bash
--region-weight > 0
```

训练入口会额外建立：

```text
region_target_key = mean_<region_hierarchy>
```

例如：

```bash
--region-weight 0.1 \
--region-hierarchy 60
```

意味着 region loss 使用：

```text
mean_60
```

作为监督目标。

因此，只要满足下面任意一个条件：

```text
A. model uses mean
B. region loss enabled
```

就必须提供：

```bash
--train-mean ...
```

---

# 8. 图像尺寸

常规训练目前主要使用：

```bash
--image-size 352
```

实验中如果专门研究输入分辨率，可以显式改成：

```bash
--image-size 512
```

但这是实验变量。

生成 ablation 脚本时不要无意中改变。

---

# 9. Epoch

例如：

```bash
--epochs 50
```

或者某些历史实验：

```bash
--epochs 45
```

注意：

```text
run-dir 名字中的 e45 / e50
```

只是名字，不会控制真正训练 epoch。

真正有效的是：

```bash
--epochs
```

因此不能根据：

```text
runs/xxx_e45
```

就认定 checkpoint 一定来自 45 epoch。

测试前如果实验对 epoch 敏感，应查看 checkpoint / log 中记录的真实 epoch。

---

# 10. Batch Size

常见：

```bash
--batch-size 8
```

显存不足时可以调整，但做结构消融时尽量保持一致。

---

# 11. DataLoader Workers

常见：

```bash
--num-workers 8
```

本地 debug 可以：

```bash
--num-workers 0
```

但正式服务器实验通常不需要设为 0。

---

# 12. Optimizer / LR 参数

常见：

```bash
--lr 1e-4
--min-lr 1e-6
--weight-decay 1e-4
```

当前常规协议使用 cosine schedule。

做网络结构消融时：

> 除非实验本身研究优化策略，否则不要顺手改 LR / scheduler / weight decay。

---

# 13. Aux Loss

常见：

```bash
--aux-weight 0.4
```

如果网络有 aux predictions，则控制 auxiliary supervision 权重。

如果某个网络没有 aux output，是否使用该参数由当前 loss / trainer 实现决定。

不要为了“保险”给网络新增 aux head。

---

# 14. Edge Loss

接口通常为：

```bash
--edge-weight 0.0
```

需要 edge supervision 的实验才设置非零。

例如：

```bash
--edge-weight 0.1
```

如果网络没有对应 edge output，不能仅靠把权重改成非零就凭空产生 edge branch。

---

# 15. Region Loss

常见接口：

```bash
--region-weight 0.0
--region-hierarchy 60
```

启用时：

```bash
--region-weight 0.1 \
--region-hierarchy 60 \
--train-mean datasets/EORSSD/train-mean
```

其中：

```text
region-hierarchy = 60
→ target = mean_60
```

---

# 16. 8-way augmentation

启用：

```bash
--augment-8way
```

关闭：

```bash
--no-augment-8way
```

或者直接不传，取决于当前入口默认值。

正式 ablation 中必须明确保持一致。

一个实验用了：

```bash
--augment-8way
```

另一个没用，就不再是单纯的网络结构对照。

---

# 17. Seed

```bash
--seed 42
```

重复实验可改：

```bash
--seed 43
--seed 44
```

但 seed 只控制显式随机源的一部分。

相同 seed 不代表 GPU 训练一定 bitwise deterministic。

因此：

```text
same seed
≠
guaranteed identical run
```

---

# 18. AMP

常规：

```bash
--amp
```

关闭：

```bash
--no-amp
```

当前 trainer 的实际精度行为应以 **当前 trainer 代码和训练日志** 为准。

不要仅根据旧实验经验假定一定是：

```text
FP16
```

或一定是：

```text
BF16
```

因为 trainer 曾经发生过数值路径变化。

做严格消融时应固定 trainer 版本。

---

# 19. Checkpoint 保存

常见：

```bash
--save-every 5
```

典型目录：

```text
run-dir/
└── checkpoints/
    ├── latest.pth
    ├── epoch_0005.pth
    ├── epoch_0010.pth
    ├── ...
    └── final.pth
```

语义：

```text
latest.pth
    每轮覆盖，用于恢复训练

epoch_xxxx.pth
    按 save-every 保存的阶段 checkpoint

final.pth
    训练结束时保存
```

---

# 20. Resume 训练

接口：

```bash
--resume runs/xxx/checkpoints/latest.pth
```

例如：

```bash
python train.py \
    --network models.networks.xxx \
    --run-dir runs/xxx \
    --resume runs/xxx/checkpoints/latest.pth \
    ...
```

resume 通常会恢复：

```text
model
optimizer
scheduler
scaler
epoch
global_step
```

## Resume 时必须遵守

### 1. network 不要换

```text
checkpoint network
=
--network
```

### 2. run-dir 通常保持原目录

否则日志、metrics、checkpoint 会分散。

### 3. 不要偷偷改训练协议

例如从 45 epoch resume 后改到 50 epoch是可以的，但必须明确知道：

```text
这是继续训练实验
```

而不是把结果误记成：

```text
独立 50e rerun
```

### 4. 训练日志要检查实际加载 epoch

不要只看 run-dir 名称。

---

# 21. Debug 小样本训练

如果当前入口支持：

```bash
--max-train-samples N
```

可用于检查：

- forward；
- loss；
- dataset；
- NAM / mean 路径；
- checkpoint；
- OOM；
- shape mismatch。

例如：

```bash
--max-train-samples 32
```

只用于 debug，不作为正式结果。

---

# 22. 日志

常见：

```bash
--log-interval 100
```

训练输出一般至少关注：

```text
Loss
Main
Aux
Grad
LR
```

某些特殊训练入口还会有自己的额外统计。

例如 SFA：

```text
BaseMain
SFA
FG ratio
FG scale
```

---

# 23. 独立训练入口：什么时候需要

不要因为网络文件变了就新建训练入口。

## 继续使用 train.py

下面这类改动通常仍应：

```bash
python train.py
```

包括：

- 换 backbone；
- 换 decoder；
- 加 attention；
- 加 fusion；
- 加 refinement；
- 加 dictionary；
- 加 region module；
- 修改 forward 内部结构；
- 新增模型输入，但已经被统一 model-input 机制支持。

即：

```text
训练协议没变
→ train.py
```

---

## 应该使用独立 train_xxx.py

如果改的是：

- loss 定义；
- supervision protocol；
- weak supervision；
- pseudo-label generation during training；
- optimizer / multi-stage schedule；
- trainer 本身；
- 特殊训练状态；
- 训练时额外 teacher / EMA / consistency branch。

则更适合：

```text
train_xxx.py
```

同时尽量：

> 复用现有 `engine.trainer`，不要无必要复制 trainer。

这样避免：

```text
网络变量
+
训练引擎变量
```

同时变化。

---

# 24. SFA / MAE-loss 分支

目前已有一个典型独立训练分支：

```bash
python train_mae_loss.py
```

它的原则是：

```text
原 train.py             不改
原 engine/trainer.py    不改
原 sod_loss.py          不改
网络                     不改
新增独立 loss
新增独立 train entry
```

例如：

```bash
python train_mae_loss.py \
    --network models.networks.mambavision_small_progressive_region_direct_hier60_sod \
    --run-dir runs/mv_progressive_region_direct_hier60_sfa_eorssd_aug8_e45 \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-mean datasets/EORSSD/train-mean \
    --image-size 352 \
    --epochs 45 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --sfa-weight 0.5 \
    --sfa-beta 0.5 \
    --sfa-gamma 2.0 \
    --sfa-max-fg-scale 8.0 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
```

专属参数：

```bash
--sfa-weight
--sfa-beta
--sfa-gamma
--sfa-max-fg-scale
```

---

# 25. 独立 loss 分支的测试

如果训练时只是改变 loss，而：

```text
network architecture 没变
checkpoint model state 格式没变
forward 没变
```

测试仍然直接：

```bash
python test.py
```

不要写一个没有必要的：

```text
test_mae_loss.py
```

例如 SFA 训练：

```text
train_mae_loss.py
        ↓
checkpoint
        ↓
test.py
```

---

# 26. 标准测试入口：test.py

典型：

```bash
python test.py \
    --network models.networks.xxx \
    --checkpoint runs/xxx/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --output-dir runs/xxx/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
```

---

# 27. Test 参数

## `--network`

必须与训练 checkpoint 对应：

```bash
--network models.networks.xxx
```

---

## `--checkpoint`

可以测试：

```text
final.pth
latest.pth
epoch_0045.pth
epoch_0050.pth
best.pth
```

只要该文件真实存在并对应当前 network。

例如：

```bash
--checkpoint runs/xxx/checkpoints/final.pth
```

不要因为文件夹名叫 `e45` 就默认：

```text
final.pth = epoch 45
```

必要时读取 checkpoint metadata 或日志确认。

---

## `--test-images`

```bash
--test-images datasets/EORSSD/test-images
```

---

## `--test-masks`

```bash
--test-masks datasets/EORSSD/test-labels
```

用于指标评估。

---

## `--output-dir`

预测结果与测试日志输出位置：

```bash
--output-dir runs/xxx/test/EORSSD
```

建议始终放在当前实验自己的 run-dir 下。

---

## `--dataset-name`

```bash
--dataset-name EORSSD
```

测试日志中用于标识数据集。

切换数据集时一起改：

```text
EORSSD
ORSSD
ORSI-4199
```

---

## `--image-size`

常规：

```bash
--image-size 352
```

如果训练和特定测试协议要求其他尺寸，则明确修改。

---

## `--batch-size`

例如：

```bash
--batch-size 8
```

推理显存不足时可以降低。

---

## `--num-workers`

例如：

```bash
--num-workers 8
```

---

## `--warmup-steps`

例如：

```bash
--warmup-steps 10
```

主要用于测速前 warmup。

---

## `--log-interval`

例如：

```bash
--log-interval 100
```

---

# 28. 测试 NAM 网络

如果网络 forward 需要 NAM，则 test 侧也必须提供对应输入。

形式与训练侧对称：

```text
train:
--train-nam ...

test:
--test-nam ...
```

具体参数名以当前 `test.py --help` 为准。

原则：

```text
训练时模型依赖什么外部输入
测试时也必须提供对应输入
```

不能训练时：

```text
RGB + NAM
```

测试时只给：

```text
RGB
```

---

# 29. 测试 Region Mean 网络

典型：

```bash
python test.py \
    --network models.networks.mambavision_small_progressive_region_direct_hier60_sod \
    --checkpoint runs/xxx/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --test-mean datasets/EORSSD/test-mean \
    --output-dir runs/xxx/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
```

和训练侧一样：

```text
model uses mean
→ test 必须有 --test-mean
```

---

# 30. Region Loss 不影响测试接口

如果某实验训练时：

```bash
--region-weight 0.1
```

但网络本身 forward 并不需要 mean：

```text
mean 只用于训练 supervision
```

那么测试不需要为了 region loss 再保留监督 target。

关键判断是：

```text
测试 forward 是否需要这个输入？
```

而不是：

```text
训练 loss 有没有用过这个输入？
```

---

# 31. Dataset 切换

同一个 checkpoint 在不同数据集上测试时，主要切：

```bash
--test-images
--test-masks
--test-nam       # 如果网络需要
--test-mean      # 如果网络需要
--output-dir
--dataset-name
```

例如：

```text
EORSSD
ORSSD
ORSI-4199
```

网络和 checkpoint 不变：

```bash
--network ...
--checkpoint ...
```

---

# 32. 一个完整的普通训练模板

```bash
#!/usr/bin/env bash

set -e
unset OMP_NUM_THREADS

cd /home/data2/MaLei/MLab

python train.py \
    --network models.networks.NETWORK_NAME \
    --run-dir runs/RUN_NAME \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --image-size 352 \
    --epochs 50 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
```

---

# 33. 一个完整的 Region Mean 训练模板

```bash
#!/usr/bin/env bash

set -e
unset OMP_NUM_THREADS

cd /home/data2/MaLei/MLab

python train.py \
    --network models.networks.NETWORK_NAME \
    --run-dir runs/RUN_NAME \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-mean datasets/EORSSD/train-mean \
    --image-size 352 \
    --epochs 50 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
```

---

# 34. 一个完整的 Region Loss 训练模板

```bash
#!/usr/bin/env bash

set -e
unset OMP_NUM_THREADS

cd /home/data2/MaLei/MLab

python train.py \
    --network models.networks.NETWORK_NAME \
    --run-dir runs/RUN_NAME \
    --train-images datasets/EORSSD/train-images \
    --train-masks datasets/EORSSD/train-labels \
    --train-mean datasets/EORSSD/train-mean \
    --image-size 352 \
    --epochs 50 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --weight-decay 1e-4 \
    --aux-weight 0.4 \
    --region-weight 0.1 \
    --region-hierarchy 60 \
    --augment-8way \
    --seed 42 \
    --save-every 5 \
    --log-interval 100
```

---

# 35. 一个完整的标准测试模板

```bash
#!/usr/bin/env bash

set -e
unset OMP_NUM_THREADS

cd /home/data2/MaLei/MLab

python test.py \
    --network models.networks.NETWORK_NAME \
    --checkpoint runs/RUN_NAME/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --output-dir runs/RUN_NAME/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
```

---

# 36. 一个完整的 Region Mean 测试模板

```bash
#!/usr/bin/env bash

set -e
unset OMP_NUM_THREADS

cd /home/data2/MaLei/MLab

python test.py \
    --network models.networks.NETWORK_NAME \
    --checkpoint runs/RUN_NAME/checkpoints/final.pth \
    --test-images datasets/EORSSD/test-images \
    --test-masks datasets/EORSSD/test-labels \
    --test-mean datasets/EORSSD/test-mean \
    --output-dir runs/RUN_NAME/test/EORSSD \
    --dataset-name EORSSD \
    --image-size 352 \
    --batch-size 8 \
    --num-workers 8 \
    --warmup-steps 10 \
    --log-interval 100
```

---

# 37. 多卡 / 选卡

学校服务器通常需要显式选择 GPU。

可以在运行前：

```bash
export CUDA_VISIBLE_DEVICES=0
```

然后：

```bash
python train.py ...
```

或者单行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py ...
```

如果使用 shell script：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_xxx.sh
```

注意：

```text
CUDA_VISIBLE_DEVICES=2
```

之后，程序内部看到的通常仍然是：

```text
cuda:0
```

因为可见设备已经被重映射。

---

# 38. Shell Script 的建议结构

统一：

```bash
#!/usr/bin/env bash

set -e

unset OMP_NUM_THREADS

cd /正确的/MLab

python train.py \
    ...
```

测试同理。

`set -e`：

```text
任意命令失败
→ shell 停止
```

避免前一步失败后继续跑错误的后续命令。

---

# 39. 脚本命名建议

训练：

```text
train_<network-short-name>_<dataset>_<special-protocol>.sh
```

测试：

```text
test_<network-short-name>_<dataset>.sh
```

例如：

```text
train_mv_progressive_region_direct_hier60_eorssd_aug8.sh
test_mv_progressive_region_direct_hier60_eorssd.sh
```

loss 分支：

```text
train_mv_progressive_region_direct_hier60_sfa_eorssd_aug8.sh
test_mv_progressive_region_direct_hier60_sfa_eorssd.sh
```

---

# 40. Run-dir 命名建议

run-dir 应尽量把真正改变实验结果的主要变量写进去。

例如：

```text
runs/
mv_progressive_region_direct_hier60_eorssd_aug8_e50
```

可包括：

```text
network family
main variant
dataset
augmentation
epoch
special loss/protocol
```

不要把没有实际启用的模块写进名字。

也不要把名字当实验事实。

最终事实以：

```text
args.json
train.log
checkpoint metadata
```

为准。

---

# 41. 测试前检查表

测试前按顺序检查：

```text
[ ] network module 是否与 checkpoint 对应
[ ] checkpoint 是否真实存在
[ ] checkpoint 的真实 epoch 是否确认
[ ] test-images 是否对应正确数据集
[ ] test-masks 是否对应正确数据集
[ ] 网络若需要 NAM，是否提供 test NAM
[ ] 网络若需要 mean，是否提供 test mean
[ ] output-dir 是否属于当前 run
[ ] dataset-name 是否正确
[ ] image-size 是否符合当前协议
```

---

# 42. 训练前检查表

```text
[ ] 当前在哪台服务器
[ ] cd 路径是否正确
[ ] 数据路径是否属于该服务器
[ ] network module 是否正确
[ ] run-dir 是否唯一
[ ] train-images / masks 是否匹配
[ ] 模型是否需要 NAM
[ ] 模型是否需要 mean
[ ] 是否启用 region loss
[ ] image-size 是否是当前实验变量
[ ] epoch 是否正确
[ ] augment-8way 是否与 baseline 一致
[ ] seed 是否是计划中的 seed
[ ] trainer / loss entry 是否是本实验计划中的版本
[ ] resume 是否真的需要
```

---

# 43. 生成新实验脚本时的决策流程

以后为 MLab 生成 train/test 脚本时，按下面顺序判断。

```text
1. 这是哪个运行环境？
   ├─ 学校服务器 CFMNet
   ├─ 学校服务器弱监督 VMamba
   └─ AutoDL 标准 VMamba

2. 网络 module 是什么？
   └─ models.networks.xxx

3. 训练协议有没有变？
   ├─ 没变 → train.py
   └─ 变了 → 对应 train_xxx.py

4. 模型需要什么输入？
   ├─ RGB
   ├─ NAM
   └─ mean

5. loss 额外需要什么监督？
   ├─ edge
   └─ region mean

6. 是否 resume？
   ├─ no
   └─ yes → 确认 checkpoint / epoch / optimizer state

7. 测试时 forward 需要什么？
   ├─ RGB
   ├─ NAM
   └─ mean

8. 训练专属 loss 是否改变模型 forward？
   ├─ no → test.py
   └─ yes → 才考虑独立 test entry
```

---

# 44. 最重要的几条规则

## Rule 1

**换网络，不等于换训练入口。**

```text
architecture changed
training protocol unchanged
→ train.py
```

---

## Rule 2

**换 loss，可以有独立 train，但通常不用独立 test。**

```text
train_mae_loss.py
→ checkpoint
→ test.py
```

---

## Rule 3

**NAM / mean 是否读取，由模型真实输入需求决定。**

不要根据网络名字猜。

---

## Rule 4

**Region Mean 输入和 Region Loss target 不要混为一谈。**

```text
mean as model input
≠
mean as training target
```

---

## Rule 5

**run-dir 名字不是实验真相。**

真实配置优先看：

```text
args.json
train.log
checkpoint metadata
```

---

## Rule 6

**resume 后的训练不是独立 rerun。**

必须明确记录：

```text
从哪个 epoch
继续到哪个 epoch
```

---

## Rule 7

**做 ablation 时固定训练引擎。**

特别注意：

```text
AMP dtype
GradScaler
gradient clipping
optimizer
scheduler
augmentation
epoch
seed
```

不要让这些变量和网络结构同时偷偷变化。

---

## Rule 8

**生成脚本前先确定服务器路径。**

尤其不要再次混用：

```text
/home/MLab
```

与：

```text
/home/data2/MaLei/MLab
```

---

# 45. 不确定接口时

如果当前 master 已经发生变化，不要凭 README 猜参数。

直接执行：

```bash
python train.py --help
python test.py --help
```

特殊入口：

```bash
python train_mae_loss.py --help
```

然后以 **当前代码** 为准。

这份 README 的定位是：

```text
接口约定 + 使用规则 + 历史上容易踩坑的地方
```

不是替代源码。

---

# 46. 给后续 ChatGPT / Codex 的约束

当需要为 MLab 生成新的训练 / 测试脚本时：

1. 优先复用现有 `train.py / test.py`。
2. 不因为新增 network 文件而复制 trainer。
3. 先确认模型真实输入，再决定是否加入 `--train/test-nam`、`--train/test-mean`。
4. 独立 loss 使用独立 train entry，但测试优先复用 `test.py`。
5. 不主动修改 trainer、AMP、gradient clipping、optimizer、scheduler。
6. 不把路径从一个服务器环境机械复制到另一个环境。
7. `run-dir`、checkpoint、network 三者必须匹配。
8. resume 实验要明确真实起止 epoch。
9. 新实验脚本尽量只改变当前需要验证的变量。
10. 如果源码与本文档冲突，以当前源码为准，并同步更新本文档。
