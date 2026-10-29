from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# tool import
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests


# creating models
# model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

llm = HuggingFaceEndpoint(
    repo_id='ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF',
    task='text-generation'
)
model = ChatHuggingFace(llm = llm)


#================================================
#   creating tools
#================================================

search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num:float, second_num:float, operation:str) -> dict:
    """
    perform basic arithmatic operation on two number
    supported operation: add,sub, mul, div
    """
    try:
        if operation == 'add':
            result = first_num + second_num
        elif operation == 'sub':
            if first_num > second_num:
                result = first_num - second_num
            else:
                result = second_num - first_num
        elif operation =='div':
            if second_num ==0:
                return {'error':'division by zero is not allowed'}
            result= first_num / second_num
        elif operation == 'mul':
            result = first_num * second_num

        else:
            return {'error':f'unsupported operation "{operation}"'}

        return {'first_num':first_num, 'second_num':second_num, 'second_num':second_num, 'result':result}
    except Exception as e:
        return {'error':str(e)}


@tool
def get_stock_price(symbol:str) -> dict:
    """
    fetch latest stock price for a given symbol (eg: 'AAPL', 'TSLA')
    using alpha vantage with api key in the url.
    """
    apikey = 'Q31AYZ4OHSD2EU2P'
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}"


    response = requests.get(url)
    return response.json()


tools = [search_tool, get_stock_price, calculator]

llm_with_tools = model.bind_tools(tools=tools)

#creating  states
class chatestate(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]



def chat_node(state:chatestate) -> chatestate:
    """
    llm node that may answer or request to a tool call.
    """
    messages = state['messages']
    response  = llm_with_tools.invoke(messages)

    return {'messages':[response]}


tool_node = ToolNode(tools) # executes tool call
print('all functions written...')

#==========================================
# checkpointer
# ===================================

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(chatestate)
print('graph initialized...')

# creating node
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)
print('node created ...')

# creating edge
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools','chat_node')
# graph.add_edge('chat_node',END)

print('graph created...')


print('compiling chatbot...')
chatbot = graph.compile(checkpointer=checkpointer)
print('chatbot compiled...')
print("\n" + "=" * 60)
print("ASCII DIAGRAM")
print("=" * 60)
print(chatbot.get_graph().draw_ascii())
print('graph compiled...')


out = chatbot.invoke(
    {'messages':[HumanMessage(content='what is 2*3 ?')]},
    config={'configurable': {'thread_id': 'default-thread'}}
)
print(out['messages'][-1].content)



#==============================
# 7.helper function
#==============================
def retrieve_all_threads():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])


    return list(all_thread)




