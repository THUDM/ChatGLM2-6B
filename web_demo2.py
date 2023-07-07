from transformers import AutoModel, AutoTokenizer
import streamlit as st


st.set_page_config(
    page_title="ChatGLM2-6b 演示",
    page_icon=":robot:",
    layout='wide'
)


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model = model.eval()
    return tokenizer, model


tokenizer, model = get_model()

st.title("ChatGLM2-6B")

max_length = st.sidebar.slider(
    'max_length', 0, 32768, 8192, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.8, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.8, step=0.01
)

if 'history' not in st.session_state:
    st.session_state.history = []

if 'past_key_values' not in st.session_state:
    st.session_state.past_key_values = None

for i, (query, response) in enumerate(st.session_state.history):
    with st.chat_message(name="user", avatar="user"):
        st.markdown(query)
    with st.chat_message(name="assistant", avatar="assistant"):
        st.markdown(response)
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.text_area(label="用户命令输入",
                           height=100,
                           placeholder="请在这儿输入您的命令")

button = st.button("发送", key="predict")

if button:
    input_placeholder.markdown(prompt_text)
    history, past_key_values = st.session_state.history, st.session_state.past_key_values
    for response, history, past_key_values in model.stream_chat(tokenizer, prompt_text, history,
                                                                past_key_values=past_key_values,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature,
                                                                return_past_key_values=True):
        message_placeholder.markdown(response)

    st.session_state.history = history
    st.session_state.past_key_values = past_key_values
