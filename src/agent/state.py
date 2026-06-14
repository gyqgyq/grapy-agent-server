import operator
from dataclasses import dataclass
from typing_extensions import TypedDict, Annotated

from langchain.messages import AnyMessage



class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


@dataclass
class MemoryContext:
    user_id: str