FROM nvcr.io/nvidia/pytorch:23.06-py3

WORKDIR /app

# RUN git clone --depth=1 https://github.com/THUDM/ChatGLM2-6B.git /app

COPY . .

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -r requirements.txt

EXPOSE 7860 8000

CMD python ${CLI_ARGS}