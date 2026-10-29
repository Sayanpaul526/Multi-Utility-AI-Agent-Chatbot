#======== imports ===========
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint ,ChatHuggingFace,HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import tool
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages  import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition

import os
os.environ["LANGSMITH_PROJECT"] = 'MY-RAG-LANGGRAPH-CHATBOT'


#===================================================
# creating model

# model = ChatGoogleGenerativeAI(model = "gemini-3.7-flash")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    # repo_id="Qwen/Qwen3-8B",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm)

print('model initialized....')
#===================================================
#pdf loader 
loader = PyPDFLoader("intro_to_ml.pdf")
docs = loader.load()

print('length of docs = ',len(docs))
print('loaded the files....')
#====================================================
# splitting the documents
splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
chunks = splitter.split_documents(docs)

print('length of chunks : ',len(chunks))
print('splitting done.....')
#=====================================================

# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
vector_store = FAISS.from_documents(chunks,embeddings)

print('embedding done....')
#=====================================================

retriever = vector_store.as_retriever(search_type = 'mmr', search_kwargs = {'k':5, 'fetch_k':20, 'lambda_mult':0.5} )
# print(retriever.invoke("what is ml ?"))
print('retrieving done....')
#========================================================

# defining tools

# tool: RAG

@tool
def rag_tool(query):
    """
    reteieve the relevent information from the pdf document.
    use this tool whent he user as factual/ conceptual questions
    that might be answeres from the stored document.

    if you dont have information just say "I dont have  info about this.. "
    """
    result = retriever.invoke(query)

    context= [doc.page_content for doc in result]
    metadata= [doc.metadata for doc in result]

    return {'query': query,
            'context':context,
            'metadata':metadata
            }
print('tool created....')


tools = [rag_tool]

llm_with_tools = model.bind_tools(tools=tools)


class chatState(TypedDict):
    messages :Annotated[list[BaseMessage],add_messages]

#======== defining functions ====================
def chat_node(state:chatState):
    messages = state['messages']

    response = llm_with_tools.invoke(messages)

    return {'messages':[response]}



#===== creating tool node=====================
tool_node = ToolNode(tools)
print('tool node created....')

#========= creating graph =====================
graph = StateGraph(chatState)
print('graph initialized ......')


# creating nodes
graph.add_node('chat_node',chat_node)
graph.add_node('tools', tool_node)

print('nodes created...')

# creating edges
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
# graph.add_edge('chat_node',END)

print('edges created .....')

chatbot = graph.compile()
print("\n" + "=" * 60)
print("ASCII DIAGRAM")
print("=" * 60)
print(chatbot.get_graph().draw_ascii())
print('graph ccompiled...')



# using  the chatbot
# print('chatbot response :\n')

result = chatbot.invoke(
    {
        'messages':[HumanMessage(
            content=(
                'what is machine learning ?'
            )
        )]
    }
)
print('chatbot response :\n',result['messages'][-1].content)