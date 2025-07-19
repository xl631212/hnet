# H-Net 场景图生成 (Scene Graph Generation)

基于 H-Net 架构的端到端场景图生成系统，能够将图像转换为结构化的场景图表示，识别图像中的对象、属性和关系。

## 🌟 特性

- **分层架构**: 基于 H-Net 的分层处理，包含对象检测层、属性识别层和关系推理层
- **动态分块**: 利用 H-Net 的动态分块机制，自适应地处理不同复杂度的图像区域
- **端到端训练**: 统一的损失函数，同时优化对象检测、属性预测和关系识别
- **结构化输出**: 生成标准的 JSON 格式场景图，包含对象和关系信息
- **可视化支持**: 提供丰富的可视化工具，直观展示预测结果

## 📋 输出格式

```json
{
  "objects": [
    {
      "id": 1,
      "class": "person",
      "confidence": 0.95,
      "attributes": ["sitting", "smiling"]
    },
    {
      "id": 2,
      "class": "dog",
      "confidence": 0.88,
      "attributes": ["brown", "small"]
    }
  ],
  "relationships": [
    {
      "subject": 1,
      "predicate": "holding",
      "object": 2,
      "confidence": 0.82
    }
  ]
}
```

## 🚀 快速开始

### 环境配置

1. **克隆项目**
```bash
cd /home/xuyingl/hnet/scene_graph_generation
```

2. **安装依赖**
```bash
pip install -r requirements.txt
# 或者使用 setup.py
pip install -e .
```

3. **安装 H-Net 核心模块**
```bash
# 确保 H-Net 核心代码可用
export PYTHONPATH="/home/xuyingl/hnet:$PYTHONPATH"
```

### 数据准备

1. **下载 Visual Genome 数据集**
```bash
python src/data/prepare_data.py --data_dir ./data --download
```

2. **预处理数据**
```bash
python src/data/prepare_data.py --data_dir ./data --preprocess
```

### 训练模型

```bash
python src/train.py --config configs/hnet_scene_graph.json
```

### 评估模型

```bash
python src/evaluate.py \
    --config configs/hnet_scene_graph.json \
    --checkpoint outputs/checkpoint_best.pth \
    --split test
```

### 演示推理

```bash
python src/demo.py \
    --config configs/hnet_scene_graph.json \
    --checkpoint outputs/checkpoint_best.pth \
    --image path/to/your/image.jpg \
    --output ./demo_output
```

## 📁 项目结构

```
scene_graph_generation/
├── configs/
│   └── hnet_scene_graph.json      # 模型配置文件
├── src/
│   ├── data/
│   │   ├── dataset.py              # 数据集类
│   │   └── prepare_data.py         # 数据预处理脚本
│   ├── models/
│   │   ├── scene_graph_hnet.py     # 主模型架构
│   │   └── losses.py               # 损失函数
│   ├── evaluation/
│   │   └── metrics.py              # 评估指标
│   ├── utils/
│   │   ├── checkpoint.py           # 检查点管理
│   │   └── logger.py               # 日志工具
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── demo.py                     # 演示脚本
├── requirements.txt                # 依赖列表
├── setup.py                        # 安装脚本
└── README.md                       # 项目说明
```

## 🏗️ 模型架构

### 整体架构

```
输入图像 → 图像编码器 → H-Net骨干网络 → 场景图解码器 → 输出
                ↓
        [对象检测层]
        [属性识别层]  
        [关系推理层]
```

### 核心组件

1. **图像编码器 (ImagePatchEmbedding)**
   - 将输入图像分割为 patches
   - 转换为特征向量序列

2. **H-Net 骨干网络**
   - 分层处理图像特征
   - 动态分块机制
   - 多尺度特征融合

3. **场景图解码器 (SceneGraphDecoder)**
   - 对象分类头
   - 属性预测头
   - 关系预测头
   - 对象存在性预测

### 损失函数

- **对象检测损失**: Focal Loss + 存在性损失
- **属性预测损失**: 多标签二分类损失
- **关系预测损失**: 多标签二分类损失
- **边界正则化损失**: 鼓励有意义的分层分块

## 📊 评估指标

### 主要指标

- **mAP (mean Average Precision)**: 各任务的平均精度
- **精确率 (Precision)**: 预测正确的比例
- **召回率 (Recall)**: 实际正例被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

### 场景图特定指标

- **图准确率**: 整个场景图的准确性
- **三元组召回率**: 正确预测的关系三元组比例

## ⚙️ 配置说明

### 模型配置

```json
{
  "model": {
    "arch_layout": [...],           // H-Net 架构布局
    "d_model": [...],               // 各层模型维度
    "image_size": 224,              // 输入图像尺寸
    "patch_size": 16,               // 图像块大小
    "num_object_classes": 150,      // 对象类别数
    "num_predicate_classes": 50,    // 谓词类别数
    "num_attribute_classes": 100,   // 属性类别数
    "max_objects_per_image": 30     // 每张图像最大对象数
  }
}
```

### 训练配置

```json
{
  "training": {
    "num_epochs": 100,              // 训练轮数
    "batch_size": 16,               // 批大小
    "learning_rate": 1e-4,          // 学习率
    "optimizer": "adamw",           // 优化器
    "lr_scheduler": "cosine",       // 学习率调度器
    "use_amp": true,                // 混合精度训练
    "max_grad_norm": 1.0            // 梯度裁剪
  }
}
```

## 🔧 高级用法

### 自定义数据集

1. **准备数据格式**
```python
# 图像元数据格式
{
    "image_id": 123,
    "width": 640,
    "height": 480,
    "file_name": "image.jpg"
}

# 场景图格式
{
    "image_id": 123,
    "objects": [...],
    "relationships": [...]
}
```

2. **修改数据集类**
```python
class CustomDataset(VisualGenomeDataset):
    def __init__(self, data_dir, split, config):
        # 自定义初始化逻辑
        pass
        
    def load_scene_graphs(self):
        # 自定义场景图加载逻辑
        pass
```

### 模型微调

```python
# 加载预训练模型
model = SceneGraphHNet(config)
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 冻结部分层
for param in model.image_encoder.parameters():
    param.requires_grad = False

# 微调特定层
for param in model.scene_graph_decoder.parameters():
    param.requires_grad = True
```

### 推理优化

```python
# 模型量化
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# TensorRT 优化（如果可用）
import torch_tensorrt
model_trt = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(1, 3, 224, 224).cuda()],
    enabled_precisions={torch.float, torch.half}
)
```

## 📈 性能基准

### 在 Visual Genome 数据集上的结果

| 指标 | 对象检测 | 属性预测 | 关系预测 | 整体 |
|------|----------|----------|----------|------|
| mAP  | 0.XX     | 0.XX     | 0.XX     | 0.XX |
| F1   | 0.XX     | 0.XX     | 0.XX     | 0.XX |

### 推理速度

- **GPU (RTX 3090)**: ~XX FPS
- **CPU (Intel i9)**: ~XX FPS
- **内存使用**: ~XX GB

## 🐛 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减小 batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **训练不收敛**
   - 检查学习率设置
   - 验证数据预处理
   - 调整损失函数权重

3. **推理速度慢**
   - 使用模型量化
   - 批处理推理
   - 考虑模型剪枝

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型输出
with torch.no_grad():
    outputs = model(sample_input)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

# 可视化注意力权重
if hasattr(model, 'attention_weights'):
    attention_viz = model.attention_weights.cpu().numpy()
    plt.imshow(attention_viz)
    plt.show()
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [H-Net](https://github.com/original-hnet-repo) - 核心架构
- [Visual Genome](https://visualgenome.org/) - 数据集
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱: your.email@example.com
- GitHub Issues: [项目Issues页面](https://github.com/your-repo/issues)

---

**注意**: 这是一个研究项目，模型性能可能因数据集和配置而异。建议在实际应用前进行充分测试。