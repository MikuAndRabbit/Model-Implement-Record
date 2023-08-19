# 简介

&emsp;&emsp;这一部分的代码用于评估模型在 VQA 数据集上推理的结果，其来源于 [GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA)。

# 项目结构

* `vqaEval.py`：实现了数据处理、评估分数计算相关的方法
* `vqaEvalDemo.py`：利用标注文件和模型预测结果文件计算评估分数

# 评估所需文件结构示例

## 模型预测结果文件

```json
{
    // 每一项均为 question_id: model_predict_answer 的键值对
    "262148000": "down",
    "262148001": "skateboarding",
    "262148002": "ramp",
    "393225000": "unknown",
    "393225001": "yes",
    "393225002": "yes",
    "393225003": "soup",
    ...
}
```

## 标注文件

```json
{
    "262148000": {
        // 此字典的 key 为 question_id
        "question_type": "none of the above",
        "multiple_choice_answer": "down",
        "answers": [
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 1
            },
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 2
            },
            {
                "answer": "at table",
                "answer_confidence": "yes",
                "answer_id": 3
            },
            {
                "answer": "skateboard",
                "answer_confidence": "yes",
                "answer_id": 4
            },
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 5
            },
            {
                "answer": "table",
                "answer_confidence": "yes",
                "answer_id": 6
            },
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 7
            },
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 8
            },
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 9
            },
            {
                "answer": "down",
                "answer_confidence": "yes",
                "answer_id": 10
            }
        ],
        "image_id": 262148,
        "answer_type": "other",
        "question_id": 262148000
    },
    {
        ...
    }
}
```
