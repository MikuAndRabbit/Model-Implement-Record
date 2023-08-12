# 简介

&emsp;&emsp;OpenAI DALL-E DVAE 的代码实现，模型主体结构代码来源于 [BEIT](https://github.com/microsoft/unilm/tree/master/beit) 的代码实现。

---

# 项目结构

* `modeling_discrete_vae.py`：模型实现
* `config.yaml`：控制模型结构超参数
* `data.py`：数据处理
* `process_log.py`：绘制训练日志中的 Loss 变化折线图
* `reconstruction.py`：使用训练好的 DVAE 模型重构输入的图像，并将重构后的图像保存下来
* `utils.py`：调整相关参数的方法
* `train.py`：训练框架
* `env.yaml`：训练时所使用的 Conda 环境，请注意其中可能包含了本项目运行不需要的包
