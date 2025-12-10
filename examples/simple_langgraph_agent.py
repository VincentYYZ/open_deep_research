import asyncio
import os
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command


def _load_env_from_repo_root():
    """Load .env from the repo root so local vLLM config is available."""
    # examples/ -> repo_root/.env
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # 不覆盖已经在环境里的变量
        if key and os.getenv(key) is None:
            os.environ[key] = value


_load_env_from_repo_root()


class SimpleState(MessagesState):
    """Minimal state that just carries the messages list."""


configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def router(
    state: SimpleState, config: RunnableConfig
) -> Command[Literal["answer", "__end__"]]:
    """Route to the answer node if there is a user question."""
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, HumanMessage) and last.content and str(last.content).strip():
        return Command(goto="answer")
    return Command(
        goto=END,
        update={"messages": [AIMessage(content="请输入一个问题再运行。")]},
    )


async def answer(state: SimpleState, config: RunnableConfig):
    """Call a chat model to answer the last user message using local vLLM."""
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

    last = state["messages"][-1]
    response = await model.ainvoke([last])
    return {"messages": [response]}


builder = StateGraph(SimpleState)
builder.add_node("router", router)
builder.add_node("answer", answer)
builder.add_edge(START, "router")
builder.add_edge("answer", END)
simple_graph = builder.compile()


async def main():
    user_input = input("你想问什么？ ")
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = await simple_graph.ainvoke(
        initial_state,
        config={"configurable": {}},
    )
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("Agent:", msg.content)


if __name__ == "__main__":
    asyncio.run(main())
