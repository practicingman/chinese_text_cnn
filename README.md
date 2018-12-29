## TextCNN Pytorch实现 中文文本分类
## 论文
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

## 参考
* https://github.com/yoonkim/CNN_sentence
* https://github.com/dennybritz/cnn-text-classification-tf
* https://github.com/Shawn1993/cnn-text-classification-pytorch

## 依赖项
* python3.5
* pytorch==1.0.0
* torchtext==0.3.1
* jieba==0.39

## 词向量
https://github.com/Embedding/Chinese-Word-Vectors<br>
（这里用的是Zhihu_QA 知乎问答训练出来的word Word2vec）
## 用法
```bash
python3 main.py -h
```

## 训练
```bash
python3 main.py
```

## 准确率
- [x] CNN-non-static 随机初始化Embedding
    >
        Batch[1800] - loss: 0.009499  acc: 100.0000%(128/128)
        Evaluation - loss: 0.000026  acc: 94.0000%(6616/7000)
        early stop by 1000 steps, acc: 94.0000%
- [x] CNN-static 使用预训练的静态词向量
    >
        Batch[1900] - loss: 0.011894  acc: 100.0000%(128/128)
        Evaluation - loss: 0.000018  acc: 95.0000%(6679/7000)
        early stop by 1000 steps, acc: 95.0000%

- [ ] CNN-multichannel
