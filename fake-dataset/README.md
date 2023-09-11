# 简介

&emsp;&emsp;该项目用于生成任意长度的图文对虚假随机数据集，生成的数据集仅能用于测试相关模型是否能够正常运行。

---

# 项目结构

* `config.py`：配置文件
* `generate.py`：随机生成数据集相关的方法
* `words.txt`：用于随机生成文本的词表，来源于 [mahavivo/english-wordlists](https://github.com/mahavivo/english-wordlists/blob/master/GRE_abridged.txt) (仅对该文件进行了单词提取，使得每一行保留一个单词)

---

# 使用

1. `git clone https://github.com/MikuAndRabbit/Model-Implement-Record.git`
2. `cd Model-Implement-Record/fake-dataset`
3. 修改配置文件 `config.py` 中的相关内容 (例如：数据保存位置等)
4. `python ./generate.py`

---
