## Gen AI Project : Text to Math Problem Slover 

# importing libraries
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import Tool,initialize_agent
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain,LLMMathChain
from langchain.agents.agent_types import AgentType

## Streamlit Web Interface
st.set_page_config(page_title="Text To MAth Problem Solver And Data Search Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing Llama")

## Groq Api key
groq_api_key = st.sidebar.text_input("Enter the Groq Api Key",type="password")

if not groq_api_key:
    st.warning("Please Enter the Groq api key")
    st.stop()

llm = ChatGroq(groq_api_key = groq_api_key, model="llama-3.3-70b-versatile")

# intalize the Tools
# wikipedia tool
wikipedia_wrap = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrap.run,
    description="A tool for searching the internt to find various information on the topics mentioned."
)

## Math tool
import numexpr as ne

def calculate(expression: str):
    try:
        return str(ne.evaluate(expression))
    except Exception as e:
        return f"Error: {str(e)}"

math_tool = Tool(
    name="Calculator",
    func=calculate,
    description="Useful for solving mathematical expressions. Input should be a valid math expression only."
)

prompt_template = """
You are an agent tasked with solving mathematical questions.

- Break the solution step-by-step
- Show clear reasoning
- Give final answer clearly

Question: {question}
Answer:
"""

prompt = PromptTemplate(input_variables=['question'],template=prompt_template)

## Combine all the tools into chain

chain = LLMChain(llm = llm,prompt = prompt)

reasoning_tool = Tool(
    name = "Reasoning",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions."
)


## intalize the agents
agent = initialize_agent(
    tools=[wikipedia_tool,math_tool,reasoning_tool],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True
)

## creating a session state
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am Math Chatbot who can answer all your maths questions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

## user question
user_question = st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

## start interaction
if st.button("Generate Response..."):
    if user_question:
        st.session_state.messages.append({"role":"user","content":user_question})
        st.chat_message("user").write(user_question)

        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = agent.run(user_question,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})

        st.write("## Response...")
        st.success(response)
    else:
        st.warning("Please enter the question.")
