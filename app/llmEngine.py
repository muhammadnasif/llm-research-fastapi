from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Union


from .ToolGenerator import tools
# from .methods import tools

# Create the tool
# search = TavilySearchAPIWrapper(tavily_api_key="")
# description = """"A search engine optimized for comprehensive, accurate, \
# and trusted results. Useful for when you need to answer questions \
# about current events or about recent information. \
# Input should be a search query. \
# If the user is asking about something that you don't know about, \
# you should probably use this tool to see if that can provide any information."""
# tavily_tool = TavilySearchResults(api_wrapper=search, description=description)

# tools = [tavily_tool]


# class AddNumbers(BaseModel):
#     """Input for the add_numbers tool."""

#     num1 : int = Field(..., description="First number to add.")
#     num2 : int = Field(..., description="Second number to add.")


# class AddNumbersTool(BaseTool):
#     """Tool that adds two numbers."""

#     name : str = "add_numbers"
#     description : str = "A tool that adds two numbers together."
#     args_schema = AddNumbers

#     def _run(self, num1: int, num2: int) -> int:
#         """Use the tool."""
#         return num1 + num2


# addNumberTool = AddNumbersTool()


global_session = {}
global session_id
session_id = "" 
# tools = [addNumberTool]

llm = ChatOpenAI(temperature=0, api_key="")
assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


# def _format_chat_history(chat_history: List[Tuple[str, str]]):
#     buffer = []
#     for human, ai in chat_history:
#         buffer.append(HumanMessage(content=human))
#         buffer.append(AIMessage(content=ai))
#     return buffer



chain = RunnableMap({
        "agent_scratchpad": lambda x: x["agent_scratchpad"],
        "chat_history": lambda x: x["chat_history"],
        "input": lambda x: x["input"],
        "session_id" : lambda x: x["session_id"]
    }) | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()


agent_chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(
                x["intermediate_steps"]),
    ) | chain

# memory = ConversationBufferMemory(
#                 return_messages=True, memory_key="chat_history")


# if session_id not in global_session:
#     global_session[session_id] = ConversationBufferMemory(
#         return_messages=True, memory_key="chat_history")

        
# memory = global_session[session_id]

# print("-----------------------------", global_session)
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#         "agent_scratchpad": lambda x: format_to_openai_function_messages(
#             x["intermediate_steps"]
#         ),
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIFunctionsAgentOutputParser()
# )


# class AgentInput(BaseModel):
#     input: str
#     session_id: str
# class Output(BaseModel):
#     output: Any


# agent_executor = AgentExecutor(
#             agent=agent_chain, tools=tools, verbose=True, memory=memory)

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
#     input_type=AgentInput
# )
