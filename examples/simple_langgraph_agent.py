"""Minimal LangGraph agent using a local vLLM model, with SQLite persistence enabled.

This demo shows:
- LangGraph core graph/state APIs: StateGraph, MessagesState, Command, START/END
- LangGraph persistence APIs: compile(checkpointer=...), get_state/aget_state, thread_id
- LangGraph streaming APIs: astream(stream_mode="values") for step-by-step progress
- Graph-level interrupts: interrupt_after=["confirm"] to pause durably between steps
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
from langgraph.types import Command

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


def confirm(state: SimpleState, config: RunnableConfig):
    """Checkpoint boundary node (doesn't change state).

    We pause execution AFTER this node by running the graph with:
    interrupt_after=["confirm"]

    This demonstrates durable pause/resume using the SQLite checkpointer.
    """
    return {}


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
builder.add_edge("confirm", "answer")
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

        # Durable execution（图级别暂停点）：
        # 我们每次都把图跑到 confirm 之后就暂停（interrupt_after=["confirm"]），
        # 所以如果你上次运行在 confirm 后退出了，这里会看到 next 里还有 "answer"。
        if snapshot is not None and snapshot.next and "answer" in snapshot.next:
            decision = input("检测到上次停在确认步骤：要继续回答吗？输入 yes/no ")
            if str(decision).strip().lower() in {"y", "yes"}:
                async for chunk in simple_graph.astream(
                    None,
                    config=config,
                    stream_mode="values",
                    durability="sync",
                    interrupt_after=["confirm"],
                ):
                    if "messages" in chunk:
                        messages = chunk["messages"]
            else:
                async for chunk in simple_graph.astream(
                    Command(
                        goto=END,
                        update={"messages": [AIMessage(content="好的，那我先不回答。")]},
                    ),
                    config=config,
                    stream_mode="values",
                    durability="sync",
                ):
                    if "messages" in chunk:
                        messages = chunk["messages"]

        while True:
            user_input = input("你想问什么？(输入 exit 退出) ")
            if not user_input.strip() or user_input.strip().lower() == "exit":
                break

            messages.append(HumanMessage(content=user_input))

            # LangGraph streaming:
            # - astream(..., stream_mode="values") 会按“步骤”产出当前 state 的快照
            # - 我们一边读流，一边更新 messages（对话历史）
            # - interrupt_after=["confirm"]：在 confirm 节点之后暂停（并把 next="answer" 存到 SQLite）
            async for chunk in simple_graph.astream(
                {"messages": messages},
                config=config,
                stream_mode="values",
                durability="sync",
                interrupt_after=["confirm"],
            ):
                if "messages" in chunk:
                    messages = chunk["messages"]

            decision = input("要现在回答吗？输入 yes/no ")
            if str(decision).strip().lower() in {"y", "yes"}:
                async for chunk in simple_graph.astream(
                    None,
                    config=config,
                    stream_mode="values",
                    durability="sync",
                    interrupt_after=["confirm"],
                ):
                    if "messages" in chunk:
                        messages = chunk["messages"]
            else:
                async for chunk in simple_graph.astream(
                    Command(
                        goto=END,
                        update={"messages": [AIMessage(content="好的，那我先不回答。")]},
                    ),
                    config=config,
                    stream_mode="values",
                    durability="sync",
                ):
                    if "messages" in chunk:
                        messages = chunk["messages"]

            # 打印本轮最新的 AI 回复
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    print("Agent:", msg.content)
                    break


if __name__ == "__main__":
    asyncio.run(main())
