## use pytorch images
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
## copy all files
COPY . .
## install tools
RUN apt update && apt install -y git gcc
## install requirements and cudatoolkit
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
pip install icetk -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
conda install cudatoolkit=11.7 -c nvidia
## expose port
EXPOSE 7860
## run
CMD [ "python3","web_demo.py" ]

## command for docker run 
## docker run --rm -it -v /path/to/chatglm2-6b-int4:/workspace/THUDM/chatglm2-6b --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -p 7860:7860 chatglm2:v1  python3 web_demo.py