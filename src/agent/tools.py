from langchain.tools import tool

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """计算 `a` 与 `b` 的乘积。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """计算 `a` 与 `b` 的和。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """计算 `a` 除以 `b` 的商。

    Args:
        a: 被除数
        b: 除数
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}