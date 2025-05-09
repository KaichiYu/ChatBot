from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st

st.set_page_config(page_title='基于streamlit的聊天机器人')

openai_api_base = st.sidebar.text_input(
    'OpenAI API Base', value='https://models.inference.ai.azure.com/'
)
openai_api_key = st.sidebar.text_input(
    'OpenAI API Key', type='password'
)

model = st.sidebar.selectbox(
    'Model', ('gpt-4o', 'gpt-o1', 'gpt-4.1')
)

temperature = st.sidebar.slider(
    'Temperature', 0.0, 2.0, value=0.6,
     step=0.1
)

message_history = StreamlitChatMessageHistory()

if not message_history or st.sidebar.button('清空聊天历史记录'):
    message_history.clear()
    message_history.add_ai_message('有什么可以帮助的吗？')

    # 存放中间步骤信息
    st.session_state.steps = {}

# 写历史记录，因为每次提问都会刷新和清空整个聊天列表
for index, msg in enumerate(message_history.messages):
    with st.chat_message(msg.type):
        # 写中间步骤内容
        for step in st.session_state.steps.get(str(index), []):
            if step[0].tool == '_Exception':
                continue
            with st.status(
                f'**{step[0].tool}**: {step[0].tool_input}',
                state='complete'
            ):
                st.write(step[0].log)
                st.write(step[1])
        # 写对话内容
        st.write(msg.content)

prompt = st.chat_input(placeholder='请输入提问内容')
if prompt:
    if not openai_api_key:
        st.info('请先输入OpenAI API Key')
        st.stop()

    st.chat_message('human').write(prompt)

    llm = ChatOpenAI(
        model = model,
        openai_api_key=openai_api_key,
        streaming=True,
        temperature = temperature,
        openai_api_base = openai_api_base
    )

    tools = [DuckDuckGoSearchRun(name='Search')]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm = llm, tools = tools
    )
    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key='chat_history',
        output_key='output',
        k=6
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent = chat_agent,
        tools = tools, 
        memory=memory, 
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    with st.chat_message('ai'):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts = False
        )
        response = executor(prompt, callbacks=[st_cb])

        st.write(response['output'])
        step_index = str(len(message_history.messages) - 1)
        st.session_state.steps[step_index] = response['intermediate_steps']