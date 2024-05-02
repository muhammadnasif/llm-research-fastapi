from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions


from .ToolGenerator import tools



global_session = {}
global session_id
session_id = "" 


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

chain = RunnableMap({
        "agent_scratchpad": lambda x: x["agent_scratchpad"],
        "chat_history": lambda x: x["chat_history"],
        "input": lambda x: x["input"]
    }) | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()


agent_chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(
                x["intermediate_steps"]),
    ) | chain
