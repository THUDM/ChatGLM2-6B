# 需要安装fastchat： pip install fschat 详细见repo：https://github.com/lm-sys/FastChat
# 加载不同的模型更改 command2 中的 --model-path 参数就可以，注意模型路径是config.json所在的文件夹

import subprocess

def execute_command(command):
    process = subprocess.Popen(command, shell=True)
    return process.pid


# 启动任务
command1 = 'nohup python -m fastchat.serve.controller >> fastchat_log.txt 2>&1 &'
process1 = execute_command(command1)

print(f"Process 1 started with PID: {process1}")

command2 = 'nohup python -m fastchat.serve.model_worker --model-path /root/ChatGLM2-6B_0/chatglm2-6b  >> fastchat_log.txt 2>&1 &'
process2 = execute_command(command2)
print(f"Process 2 started with PID: {process2}")

command3 = 'nohup python -m fastchat.serve.openai_api_server --host "0.0.0.0" --port 8000 >> fastchat_log.txt 2>&1 &'
process3 = execute_command(command3)
print(f"Process 3 started with PID: {process3}")





# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://0.0.0.0:8000/v1"

# model = "chatglm2-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
