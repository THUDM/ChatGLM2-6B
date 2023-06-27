首先从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0) 下载处理好的 C-Eval 数据集，解压到 `evaluation` 目录下。然后运行

```shell
cd evaluation
python evaluate_ceval.py
```

这个脚本会在C-Eval的验证集上进行预测并输出准确率。如果想要得到测试集上的结果可以将代码中的 `./CEval/val/**/*.jsonl` 改为 `./CEval/test/**/*.jsonl`，并按照 C-Eval 规定的格式保存结果并在 [官网](https://cevalbenchmark.com/) 上提交。

汇报的结果使用的是内部的并行测试框架，结果可能会有轻微波动。