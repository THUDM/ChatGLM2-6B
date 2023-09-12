# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


import time
import torch
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from pydantic import BaseModel, Field
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from transformers import AutoTokenizer, AutoModel
from typing import Any, Dict, List, Literal, Optional, Union
from typing import Union

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

'''
用户数据（模拟数据库用户表）；用于我们稍后验证。
账号：johndoe 密码：secret

为了数据安全，我们利用PassLib对入库的用户密码进行加密处理，推荐的加密算法是"Bcrypt"
其中，我们主要使用下面方法：
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pwd_context.hash(password) # 获取对密码进行加密的密文
pwd_context.verify(password, hashed_password) # 对密码进行校验

获取api-key 方法见：
官方文档: https://fastapi.tiangolo.com/zh/tutorial/security/oauth2-jwt/#_5
Postman: https://blog.csdn.net/Disany/article/details/109365066
'''
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        # 得到这串加密后的密文，请运行:
        # pwd_context.hash(password)
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


# Context是上下文,CryptContext是密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

'''
OAuth2PasswordBearer是接收URL作为参数的一个类：客户端会向该URL发送username和password参数，然后得到一个token值。
OAuth2PasswordBearer并不会创建相应的URL路径操作，只是指明了客户端用来获取token的目标URL。
当请求到来的时候，FastAPI会检查请求的Authorization头信息，如果没有找到Authorization头信息，或者头信息的内容不是Bearer token，它会返回401状态码(UNAUTHORIZED)。
'''
# oauth2_scheme是令牌对象，token: str = Depends(oauth2_scheme)后就是之前加密的令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


# verify_password验证密码
# plain_password普通密码, hashed_password哈希密码
# 返回True和False
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# 获取哈希密码;普通密码进去，对应的哈希密码出来。
def get_password_hash(password):
    return pwd_context.hash(password)


# 模拟从数据库读取用户信息
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


# 验证用户
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


# 创建访问令牌（token）
def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


'''数据请求验证
用户拿到token信息后，必须在后续请求中，头信息的Authorization带有Bearer token，才能访问其他数据接口。
下面添加一个校验函数，对请求的合法性进行校验，读取token内容解析并进行验证，验证token通过后，获取接口响应数据
'''


# 获取当前用户
# 通过oauth2_scheme，拿到用户请求头文件里的token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # jwt 解码
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # 通常在jwt 解码会进行验证抛出各种异常PyJWTError，如令牌过期等；
        # 获取生成token时候，我们放进去的username信息
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    # 获取该用户信息
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


# 获取当前激活用户，通过数据库信息及相关条件对用户有效性进行过滤；如该用户存在，密码正确，token验证通过，但数据库字段显示该用户被封号或欠费了（非激活用户），就这此处触发异常，结束访问。
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


''' ----  登录验证，获取token的接口 --------
        用户发送post请求获取token,后端验证该用户是否存在，密码是否正确。如果验证通过，会生成‘token’给到用户。
'''


# name = johndoe    password = secret
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # 1、验证用户
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 2、access_token_expires访问令牌过期
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # 3、create_access_token创建访问令牌
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/v1/models", response_model=ModelList)
# 如果不需要apikey功能，可将括号中传参的 current_user 变量删除，既可允许任意访问，无需api-key身份认证。如下示例：
# async def list_models():
async def list_models(current_user: User = Depends(get_current_active_user)):
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
# 如果不需要apikey功能，可将括号中传参的 current_user 变量删除，既可允许任意访问，无需api-key身份认证。如下示例：
# async def create_chat_completion(request: ChatCompletionRequest):
async def create_chat_completion(request: ChatCompletionRequest, current_user: User = Depends(get_current_active_user)):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i + 1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i + 1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
