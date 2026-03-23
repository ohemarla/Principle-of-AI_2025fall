# ViT 模型解析

本文结合 [ViT.py](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py) 的实际实现，说明这个 Vision Transformer 是如何完成花卉分类任务的。

## 1. 整体思路

和 `ResNet` 依赖卷积逐层提取局部特征不同，`ViT` 的核心想法是：

1. 把一张图片切成很多固定大小的小块 `patch`。
2. 把每个 `patch` 映射成一个向量，视作一个“词元”。
3. 把这些词元送入 Transformer 编码器，让模型通过自注意力学习不同区域之间的关系。
4. 用一个额外的分类 token，也就是 `cls token`，汇总全局信息并输出最终类别。

在这份作业里，输入图片大小是 `224 x 224`，类别数是 `5`，分别对应花卉分类任务中的 5 个类别。

## 2. 输入到输出的完整流程

以一张输入图片 `x` 为例，它的形状是：

```text
[batch_size, 3, 224, 224]
```

模型从输入到输出的流程如下：

1. `PatchEmbedding` 把图片切成 patch，并投影到高维向量空间。
2. 在 patch 序列前拼接一个 `cls token`。
3. 加上可学习的位置编码 `pos_embed`。
4. 经过多层 `TransformerEncoderBlock`。
5. 取第一个位置上的 `cls token` 作为整张图像的表示。
6. 送入最后的线性分类层，输出 5 个类别的 logits。

最终输出形状为：

```text
[batch_size, 5]
```

## 3. PatchEmbedding 是怎么工作的

对应代码在 [ViT.py#L10](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L10) 到 [ViT.py#L31](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L31)。

这里设置了：

- `image_size = 224`
- `patch_size = 16`
- `embed_dim = 384`

因为：

```text
224 / 16 = 14
```

所以整张图像会被切成：

```text
14 x 14 = 196
```

个 patch。

实现上并没有手动 `for` 循环切块，而是用了一个卷积层：

```python
self.proj = nn.Conv2d(
    in_channels,
    embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

这层卷积的作用可以理解为：

- 卷积核大小就是 patch 大小 `16 x 16`
- 步长也是 `16`
- 因此每滑动一次就正好对应一个 patch
- 每个 patch 最后被映射成一个长度为 `384` 的向量

输入输出形状变化如下：

```text
输入:  [B, 3, 224, 224]
卷积后: [B, 384, 14, 14]
flatten + transpose 后: [B, 196, 384]
```

此时就得到了长度为 196 的 patch 序列，每个 patch 都有一个 384 维表示。

## 4. cls token 和位置编码

对应代码在 [ViT.py#L81](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L81) 到 [ViT.py#L147](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L147)。

### 4.1 cls token

模型定义了一个可学习参数：

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
```

它的作用是作为整张图片的“汇总位”。在前向传播时，会复制成每个 batch 一份：

```text
[1, 1, 384] -> [B, 1, 384]
```

然后和 patch 序列拼接：

```text
[B, 1, 384] + [B, 196, 384] -> [B, 197, 384]
```

这样 Transformer 在后续计算时，就可以让 `cls token` 和所有 patch 交互，最终聚合全局信息。

### 4.2 位置编码

如果只把 patch 当作普通序列送进 Transformer，模型并不知道“哪个 patch 在左上角，哪个 patch 在中间”。因此需要加入位置信息：

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

这里的 `+1` 就是给 `cls token` 也分配一个位置编码。

加入位置编码后，序列形状不变：

```text
[B, 197, 384]
```

但每个 token 额外携带了自己的空间位置信息。

## 5. TransformerEncoderBlock 的工作方式

对应代码在 [ViT.py#L56](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L56) 到 [ViT.py#L78](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L78)。

每一个编码器块都包含两部分：

1. 多头自注意力 `Multi-Head Self-Attention`
2. 前馈网络 `MLP`

并且两部分都带有残差连接。

这个块的计算过程是：

```text
x -> LayerNorm -> MultiHeadAttention -> Dropout -> 残差相加
  -> LayerNorm -> MLP -> 残差相加
```

代码里写成：

```python
attn_input = self.norm1(x)
attn_output, _ = self.attn(attn_input, attn_input, attn_input)
x = x + self.dropout(attn_output)
x = x + self.mlp(self.norm2(x))
```

### 5.1 自注意力在这里做了什么

自注意力的关键价值是：每个 patch 不再只看自己附近，而是可以直接和所有 patch 建立联系。

例如一张花朵图像中：

- 中心区域可能是花瓣
- 边缘区域可能是叶子
- 某些颜色和纹理只看局部不够，但结合全图更容易判断类别

自注意力能让模型自动学习这些跨区域依赖关系。

### 5.2 为什么要多头

代码里设置：

- `embed_dim = 384`
- `num_heads = 6`

这意味着注意力会分成 6 个头并行计算。可以理解为不同的头会关注不同的模式，例如：

- 某些头更关注颜色
- 某些头更关注边缘纹理
- 某些头更关注长距离关系

### 5.3 MLP 的作用

注意力层擅长建模 token 之间的关系，但它本身的非线性表达能力有限，因此后面会接一个前馈网络 `MLP`，进一步增强特征表达。

在本实现里，MLP 的隐藏层宽度是：

```text
hidden_dim = embed_dim * mlp_ratio = 384 * 4 = 1536
```

对应代码在 [ViT.py#L34](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L34) 到 [ViT.py#L53](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L34)。

计算过程是：

```text
384 -> 1536 -> GELU -> Dropout -> 384
```

## 6. 整个 ViT 结构配置

本作业中使用的 ViT 结构参数如下：

- `image_size = 224`
- `patch_size = 16`
- `embed_dim = 384`
- `depth = 6`
- `num_heads = 6`
- `mlp_ratio = 4.0`
- `dropout = 0.1`
- `num_classes = 5`

这是一版相对轻量的 ViT，而不是特别大的标准 ViT-B/16。这样设计更适合当前作业数据集，原因是：

- 花卉数据集规模不算大
- 模型太大容易过拟合
- 训练资源通常也比大规模预训练任务更有限

## 7. 最后的分类头是怎么输出结果的

在经过 6 层 Transformer 编码器后，特征形状仍然是：

```text
[B, 197, 384]
```

之后代码会执行：

```python
x = self.norm(x)
x = x[:, 0]
x = self.head(x)
```

这里的含义是：

1. 对所有 token 再做一次 `LayerNorm`
2. 取出第 0 个 token，也就是 `cls token`
3. 通过线性层 `384 -> 5`

于是得到每个类别的分数：

```text
[B, 5]
```

训练时再把这个输出送入交叉熵损失函数：

```python
criterion = nn.CrossEntropyLoss()
```

## 8. 权重初始化

对应代码在 [ViT.py#L123](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L123) 到 [ViT.py#L134](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L134)。

初始化方式包括：

- `cls_token` 和 `pos_embed` 使用 `trunc_normal_`
- `Linear` 层权重使用 `trunc_normal_`
- `Linear` 层偏置初始化为 0
- `Conv2d` 使用 `kaiming_normal_`

这样做的目的是让模型在训练初期更稳定，避免一开始数值分布过大或过小。

## 9. 数据预处理与增强

对应代码在 [ViT.py#L150](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L150) 到 [ViT.py#L185](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L150)。

### 9.1 训练集增强

训练时使用了这些增强操作：

- `Resize(256)`
- `RandomResizedCrop(224)`
- `RandomHorizontalFlip()`
- `RandomRotation(15)`
- `ColorJitter(...)`
- `ToTensor()`
- `Normalize(...)`
- `RandomErasing(...)`

它们的作用是：

- 增强样本多样性
- 提高模型泛化能力
- 减少过拟合

### 9.2 验证集和测试集处理

验证和测试时不做随机增强，只做：

- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(...)`

这样可以保证评估过程稳定、可重复。

## 10. 训练流程

对应代码在 [ViT.py#L210](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L210) 到 [ViT.py#L274](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L274)。

训练过程如下：

1. 构建 `ViT` 模型。
2. 使用 `AdamW` 优化器。
3. 使用 `CosineAnnealingLR` 调整学习率。
4. 每个 epoch 在训练集上更新参数。
5. 每轮结束后在验证集上评估。
6. 如果验证准确率更高，就保存当前模型权重。
7. 如果连续若干轮没有提升，就早停。

这里选择 `AdamW` 而不是普通 `SGD`，主要是因为 Transformer 类模型通常更适合配合 `AdamW` 训练。

本实现中的主要训练超参数为：

- `batch_size = 32`
- `lr = 0.0003`
- `epochs = 100`
- `patience = 15`
- `weight_decay = 0.05`

## 11. Evaluate 函数做了什么

对应代码在 [ViT.py#L188](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L188) 到 [ViT.py#L207](/Users/yaoxianglin/Desktop/人工智能原理/Principle-of-AI_2025fall/HW2/ViT.py#L188)。

评估时：

1. 调用 `model.eval()` 切换到评估模式。
2. 关闭梯度计算 `torch.no_grad()`。
3. 前向计算得到输出 logits。
4. 用交叉熵损失计算平均损失。
5. 用 `torch.max` 找预测类别。
6. 统计准确率。

这样最终返回两个指标：

- `accuracy`
- `avg_loss`

## 12. 和 ResNet 的主要区别

如果和 `HW2/Resnet.py` 对比，可以看到两者的区别主要在特征提取方式上。

### ResNet

- 通过卷积层提取局部空间特征
- 通过残差结构堆叠深层网络
- 具有较强的局部归纳偏置

### ViT

- 先切 patch，再把图像当作序列处理
- 通过自注意力建模全局关系
- 对大规模数据和预训练通常更有优势

对于这份花卉作业来说，ViT 的潜在优点是：

- 更容易捕捉整朵花不同区域之间的全局关系
- 不局限于卷积局部感受野

但它也有明显挑战：

- 对数据量更敏感
- 如果没有预训练，小数据集上不一定天然比 ResNet 更强
- 训练成本通常更高

## 13. 这份实现为什么是“轻量版 ViT”

这份代码没有直接使用大规模预训练模型，而是从零开始训练一个较小的 ViT，因此做了几处控制：

- `embed_dim` 选为 `384`，不是更大的 `768`
- `depth` 只有 `6` 层
- `num_heads` 为 `6`

这样做的好处是：

- 参数量更小
- 更适合作业环境
- 显存压力更低
- 从零训练时更实际

## 14. 可以如何继续优化

如果后续想进一步提升效果，可以考虑这些方向：

1. 使用预训练 ViT 权重，再针对花卉数据集微调。
2. 增加训练轮数，并结合更细致的学习率调度。
3. 适当调整 `patch_size`、`embed_dim` 和 `depth`。
4. 加入 `label smoothing` 或更强的数据增强策略。
5. 和 `ResNet` 做系统对比，分析两者在混淆类别上的差异。

## 15. 一句话总结

这份 `ViT.py` 的核心逻辑，就是先把图像切成 patch 序列，再利用 Transformer 编码器通过自注意力学习全局信息，最后用 `cls token` 完成 5 类花卉分类。
