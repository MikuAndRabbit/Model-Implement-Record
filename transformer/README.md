# 简介

<div style="text-align: center; font-size: 17px; background-color: white;"><font style="color: red; font-size: 22px; font-weight: bold;">&#8252;</font> 该项目所实现的 MultiHeadAttention 和 Transformer 等均未经过充分的测试，如遇到问题还烦请提交 Issue 或 Pull Request，万分感谢。</div>

&emsp;&emsp;参照 PyTorch 官方实现的[注意力](https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/torch/nn/modules/activation.py#L889)和 [Transformer](https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/modules/transformer.py)，实现了使用 `torch.nn.Linear` 进行注意力计算的 `MultiHeadAttention` 和基于此的 `Transformer`。由于 PyTorch 官方实现的注意力计算没有使用 `torch.nn.Linear`，这使得我们无法使用 [loralib](https://github.com/microsoft/LoRA) 对其进行替换，进而阻止了对使用 PyTorch 官方实现的 Transformer 的模型的 LoRA 微调。这是我使用 `torch.nn.Linear` 重新实现 `MultiHeadAttention` 和基于此的 `Transformer` 的原因。

---

# 项目结构

* `utils.py`：一些工具方法
* `attention.py`：多头注意力
* `encoder.py`：TransformerEncoderLayer & TransformerEncoder
* `decoder.py`：TransformerDecoderLayer & TransformerDecoder
* `transformer.py`：Transformer
