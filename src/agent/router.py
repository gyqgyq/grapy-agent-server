import asyncio

from fastapi import APIRouter
from pydantic import BaseModel
from langchain.messages import HumanMessage

from src.agent.graph import agent

router = APIRouter(prefix="/agent", tags=["agent"])

# 请求体模型
class QueryRequest(BaseModel):
    question: str

@router.get("/health")
async def health():
    """健康检查接口"""
    return {
        "success": True,
        "message": "OK"
    }

@router.post("/chat")
async def chat(req: QueryRequest):
    """计算器智能体对话接口"""
    # 组装用户消息
    inputs = {
        "messages": [HumanMessage(content=req.question)],
        "llm_calls": 0
    }
    print(inputs)
    # 执行智能体
    result =  await asyncio.to_thread(agent.invoke, inputs)
    print(result)
    # 格式化返回（只对外暴露可读内容）
    resp_messages = []
    for msg in result["messages"]:
        resp_messages.append({
            "type": msg.type,
            "content": str(msg.content)
        })
    return {
        "success": True,
        "llm_calls": result["llm_calls"],
        "messages": resp_messages
    }