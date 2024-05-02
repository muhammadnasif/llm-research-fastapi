from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from .llmEngine import agent_chain, global_session
from .ToolGenerator import tools
import json
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from typing import Any, List, Union
import logging
from logging.handlers import TimedRotatingFileHandler




# Configure logger for HTTP request logs
http_logger = logging.getLogger('http_logger')
http_logger.setLevel(logging.INFO)
http_handler = TimedRotatingFileHandler('/var/log/nginx/llm_access.log')
http_logger.addHandler(http_handler)
http_format = logging.Formatter('%(asctime)s - %(levelname)s - HTTP Request: %(message)s')
http_handler.setFormatter(http_format)

# Configure logger for other logs
conversation_logger = logging.getLogger('conversation_log')
conversation_logger.setLevel(logging.INFO)
conversation_log_handler = TimedRotatingFileHandler('/var/log/nginx/llm_conversation.log', when='midnight', interval=1, backupCount=7)
conversation_logger.addHandler(conversation_log_handler)
other_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
conversation_log_handler.setFormatter(other_format)

class LlmRequest(BaseModel):
    input: str
    session_id: str

class AgentInput(BaseModel):
    input: str
    session_id: str
class Output(BaseModel):
    output: Any

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/v1/llm-engine")
def llm_engine(request : LlmRequest):
    question = request.input
    session = request.session_id


    if session not in global_session:
        global_session[session] = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history")

        
    memory = global_session[session]


    agent_executor = AgentExecutor(
            agent=agent_chain, tools=tools, verbose=True, memory=memory)
    
    llm_response = agent_executor.with_types(input_type=AgentInput, output_type=Output).invoke({"input": question})
    
    # logging.basicConfig(filename='/var/log/nginx/llm_access.log', level=logging.INFO)
    # logging.info(f"Session: {session} - Input: {question} - Output: {llm_response['output']} - Conversation Data {memory.chat_memory.messages}")    

    conversation_logger.info(f"Session: {session} - Input: {question} - Output: {llm_response['output']} - Conversation Data {memory.chat_memory.messages}")
    

    if 'function-name' in llm_response['output']:
        function_info = json.loads(llm_response['output'])
        answer = None
    else:
        function_info = None
        answer = llm_response['output']

    response_data = {
        "success": True,
        "message": "Response received successfully",
        "function-call-status": True if 'function-name' in llm_response['output'] else False,
        "data": {
            "query": question,
            "answer": answer
        },
        "function": function_info
    }
    global_session[session].chat_memory.messages = memory.chat_memory.messages[-14:]

    return response_data
# add_routes(app, rag_chroma_chain, path="/rag-chroma")
# add_routes(app, agent_executor, path="/llm-engine")

# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
