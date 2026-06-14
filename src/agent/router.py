from fastapi import APIRouter, Request
from pydantic import BaseModel
from langchain.messages import HumanMessage

from src.agent.state import MemoryContext

router = APIRouter(prefix="/agent", tags=["agent"])

# 请求体模型
class QueryRequest(BaseModel):
    question: str
    thread_id: str
    user_id: str

@router.get("/health")
async def health():
    """健康检查接口"""
    return {
        "success": True,
        "message": "OK"
    }

@router.post("/chat")
async def chat(req: QueryRequest, request: Request):
    """计算器智能体对话接口"""
    inputs = {
        "messages": [HumanMessage(content=req.question)],
    }
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await request.app.state.agent.ainvoke(
        inputs,
        config,
        context=MemoryContext(user_id=req.user_id),
    )
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