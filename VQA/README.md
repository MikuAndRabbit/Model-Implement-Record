# 项目内容

## 简介

&emsp;&emsp;本项目主要完成基于预训练模型的 VQA 下游任务的模型 (VQA Head)、训练、评估、推理的代码实现。具体来说，本项目的 VQA 下游任务是基于一个在 MSCOCO 数据上进行三种训练任务 (ITM、MLM、MIM) 的预训练模型开展的。

&emsp;&emsp;本项目实现的 VQA 并不局限于某一个预训练模型，只要有上游模型能够提供 `text_embedding` 和 `image_embedding`，我们的实现就能够进行 VQA 任务的训练、评估和推理。

## VQA 的实现思路

&emsp;&emsp;经过对现有方法的调研，大部分实现 VQA 的思路是将 VQA 转化为一个**多标签的分类任务**，其中标签的个数即为候选答案的个数。我采用的也是这种思路来完成 VQA 任务。

&emsp;&emsp;具体来说，使用将预训练模型的 Text Embedding 和 Image Embedding 元素相乘后的结果作为 VQA Head 的输入。VQA Head 一般为若干全连接层，最后一个全连接层将张量映射到 $\R^{n}$ 空间中，其中 $n$ 为候选答案的个数。

## 两种 VQA Head

&emsp;&emsp;本项目实现了两种 VQA Head，他们的具体代码实现可见下方代码或者 [vqa.py](./vqa.py) 文件。

### Plain Head

```python
class Plain_VQA_Head(nn.Module):
    def __init__(self, input_dim: int, num_answers: int, layernorm_eps: float = 1e-12) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_answers = num_answers
        self.hidden_dim = input_dim * 2
        self.layernorm_eps = layernorm_eps
        
        self.vqa_classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim, eps = self.layernorm_eps),
            nn.Linear(self.hidden_dim, self.num_answers)
        )

    
    def forward(self, fusion_feature):
        return self.vqa_classifier(fusion_feature)
```

### BEiT Head

```python
class BEiT3_VQA_Head(nn.Module):
    def __init__(self, input_dim: int, num_answers: int, layernorm_eps: float = 1e-12) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.answer_num = num_answers
        self.pooler = Pooler(
            input_features = input_dim, 
            output_features = input_dim, 
            norm_layer = nn.LayerNorm, 
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), 
            nn.LayerNorm(input_dim * 2, eps = layernorm_eps), 
            nn.GELU(), 
            nn.Linear(input_dim * 2, num_answers), 
        )


    def forward(self, fusion_feature):
        representation = self.pooler(fusion_feature)
        return self.head(representation)
```

---

# 项目结构

```
VQA
├── README.md
├── config
│   ├── config.yml: VQA 模型、训练、验证、评估的相关参数
│   ├── inference-config.yml: 推理时所使用的配置文件
│   ├── lora.py: 加载 BERT 与 BiT 的 LoRA Config
│   ├── lora.yaml: LoRA 相关参数配置文件
│   ├── tmep-config.yml: backbone model 的相关参数
│   └── yaml_config.py: 读取 yaml 文件的相关方法
├── data
│   ├── dataset.py: VQA2Dataset
│   ├── example: 模型加载训练、验证数据时所用的标准格式文件
│   │   └── question_answer_image.json
│   └── vqa2_preprocess.py: 预处理 VQA-v2 数据集
├── eval.py: VQA 评估
├── inference.py: VQA 推理
├── model: backbone model 的代码
│   ├── Discrete_vae.py
│   ├── TMEP.py
│   ├── TMEP_pretrain.py
│   └── TVA_Transformer.py
├── pretrain_weights: 存放加载模型所需的权重文件
├── train.py: 模型训练框架
├── vocab
│   └── en_vocab.txt: BertTokenizer 所使用的词表
├── vqa.py: VQA 下游任务相关的模型
└── vqa2_eval_tools: VQA-v2 数据集官方提供的评估代码 (升级到 python3 可用版本)
    ├── README.md
    ├── vqaEval.py
    └── vqaEvalDemo.py
```

---

# Tips

&emsp;&emsp;本项目 VQA 使用了在 MSCOCO 上进行预训练的 backbone model 的中间输出 `cross_text_embedding` 和 `cross_image_embedding`。二者的形状和预训练模型输入的结构如下所示：

```
Input:
    image: [batch_size, 3, 224, 224]
    question_token: [batch_size, 1, 128]
    question_attention_mask: [batch_size, 1, 128]
Output:
    cross_text_embedding: [batch_size, 128, 768]
    cross_image_embedding: [batch_size, 197, 768] (197 = patch_num + 1)
```

---
