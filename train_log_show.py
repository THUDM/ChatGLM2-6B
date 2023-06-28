import json
import matplotlib.pyplot as plt

# 从文件读取日志数据
with open("../trainer_log.jsonl", "r") as f:
    logs = [json.loads(line) for line in f.readlines()]

# 提取关键信息
steps = [log["current_steps"] for log in logs]
loss = [log["loss"] for log in logs]
learning_rate = [log["learning_rate"] for log in logs]

# 创建一个新的图形
plt.figure()

# 画出损失曲线
plt.subplot(211)
plt.plot(steps, loss, label='Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()

# 画出学习率曲线
plt.subplot(212)
plt.plot(steps, learning_rate, label='Learning Rate')
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.legend()

# 显示图形
plt.show()
