"""Minimal LangGraph agent using a local vLLM model, with SQLite persistence enabled.

This demo shows:
- LangGraph core graph/state APIs: StateGraph, MessagesState, Command, START/END
- LangGraph persistence APIs: compile(checkpointer=...), get_state/aget_state, thread_id
- Using a local OpenAI-compatible model (e.g. vLLM) via OPENAI_BASE_URL
"""

import asyncio
import os
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# LangGraph: 核心图与状态模块
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

# LangGraph persistence: 使用异步 SQLite 检查点后端
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def _load_env_from_repo_root() -> None:
    """Load .env from the repo root so local vLLM config is available."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and os.getenv(key) is None:
            os.environ[key] = value 


_load_env_from_repo_root()


class SimpleState(MessagesState):
    """Minimal state that just carries the messages list."""


# LangGraph + LangChain: 可配置模型句柄
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


# LangGraph persistence: SQLite 检查点路径
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "simple_agent_checkpoints.sqlite"


async def router(
    state: SimpleState, config: RunnableConfig
) -> Command[Literal["confirm", "__end__"]]:
    """Route to the confirm node if there is a user question."""
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, HumanMessage) and str(last.content).strip():
        return Command(goto="confirm")
    return Command(
        goto=END,
        update={"messages": [AIMessage(content="请输入一个问题再运行。")]},
    )


async def confirm(
    state: SimpleState, config: RunnableConfig
) -> Command[Literal["answer", "__end__"]]:
    """Durable execution 小例子：在这里“停下来”等你输入。

    interrupt(...) 会让图执行到这里时暂停，并把暂停点和当前状态
    存到 SQLite 检查点里。之后我们可以用 Command(resume=...)
    从这个节点继续执行。
    """
    decision = interrupt("要现在回答吗？输入 yes/no")
    if str(decision).strip().lower() in {"y", "yes"}:
        return Command(goto="answer")
    return Command(
        goto=END,
        update={"messages": [AIMessage(content="好的，那我先不回答。")]},
    )


async def answer(state: SimpleState, config: RunnableConfig):
    """Call a chat model using full conversation history and a local vLLM endpoint."""
    configurable = config.get("configurable", {}) if config else {}

    # 默认使用你本地 vLLM 暴露的模型
    model_name = configurable.get(
        "model", "openai:mistralai/Ministral-3-14B-Reasoning-2512"
    )
    max_tokens = configurable.get("max_tokens", 256)

    # 优先用 config 里的 api_key，否则用环境变量；如果设置了 OPENAI_BASE_URL 但没有显式 key，则用 dummy
    api_key = configurable.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key and os.getenv("OPENAI_BASE_URL"):
        api_key = "dummy"

    model = configurable_model.with_config(
        {
            "model": model_name,
            "max_tokens": max_tokens,
            "api_key": api_key,
        }
    )

    # 使用整个对话历史，而不是只看最后一句
    history = state["messages"]
    system_msg = SystemMessage(
        content="你是一个对话助手，会结合整个历史对话来回答用户当前的问题。"
    )
    response = await model.ainvoke([system_msg, *history])
    return {"messages": [response]}


# LangGraph: 构建简单的图（router -> confirm -> answer）
builder = StateGraph(SimpleState)
builder.add_node("router", router)
builder.add_node("confirm", confirm)
builder.add_node("answer", answer)
builder.add_edge(START, "router")
builder.add_edge("router", "confirm")
builder.add_edge("router", END)
builder.add_edge("confirm", "answer")
builder.add_edge("confirm", END)
builder.add_edge("answer", END)


async def main() -> None:
    """Simple REPL that keeps conversation history across runs using SQLite persistence."""
    # LangGraph persistence: 使用异步 SQLite 检查点，需要在 async with 上下文中创建 AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as checkpointer:
        # LangGraph persistence: 将检查点后端挂到图上
        simple_graph = builder.compile(checkpointer=checkpointer)

        # LangGraph persistence: 使用 thread_id 区分不同会话
        thread_id = "simple-langgraph-agent-demo"
        config = {"configurable": {"thread_id": thread_id}}

        # LangGraph persistence: 启动时尝试从检查点恢复历史消息（异步版本）
        messages: list = []
        snapshot = await simple_graph.aget_state(config)
        if snapshot is not None and snapshot.values:
            existing = snapshot.values.get("messages")
            if isinstance(existing, list):
                messages = existing

        # Durable execution 的关键效果之一：
        # 如果上次运行在 confirm 节点 interrupt 处中断了（比如你关掉程序），
        # 这里会检测到未完成的 interrupt，并让你输入后继续执行。
        if snapshot is not None and snapshot.interrupts:
            pending = snapshot.interrupts[0]
            resume_val = input(f"检测到上次执行暂停：{pending.value} ")
            result = await simple_graph.ainvoke(
                Command(resume=resume_val),
                config=config,
                durability="sync",
            )
            messages = result.get("messages", messages)

        while True:
            user_input = input("你想问什么？(输入 exit 退出) ")
            if not user_input.strip() or user_input.strip().lower() == "exit":
                break

            messages.append(HumanMessage(content=user_input))
            result = await simple_graph.ainvoke(
                {"messages": messages},
                config=config,
                durability="sync",
            )

            # 如果执行到 confirm 节点触发了 interrupt，就在这里读取用户输入并恢复执行
            if "__interrupt__" in result:
                interrupts = result.get("__interrupt__") or []
                prompt = interrupts[0].value if interrupts else "请输入："
                resume_val = input(f"{prompt} ")
                messages = result.get("messages", messages)
                result = await simple_graph.ainvoke(
                    Command(resume=resume_val),
                    config=config,
                    durability="sync",
                )

            messages = result["messages"]

            # 打印本轮最新的 AI 回复
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    print("Agent:", msg.content)
                    break


if __name__ == "__main__":
    asyncio.run(main())
