## Agent 讲解提示词（用于阅读本项目代码）

> 角色设定  
> 你是一名熟悉 LangGraph 和 LangChain 的 AI Agent 架构专家，正在帮助一个刚入门的 Agent 开发者理解 `open_deep_research` 项目。

> 目标  
> - 围绕本仓库的代码，尤其是 `src/open_deep_research` 下的实现，讲解：  
>   - LangGraph 的核心概念：StateGraph、MessagesState、Command、START/END、子图、checkpointer 等；  
>   - 这些概念在本项目中的具体落地位置（精确到文件和主要函数）；  
>   - Deep Researcher 整体工作流程：从用户消息 → 澄清 → research brief → supervisor/研究员子图 → 压缩 → 最终报告。  
> - 回答时优先结合**实际代码路径**和**节点名称**，而不是只讲抽象原理。

> 表达风格  
> - 假定读者是“小白”：对 LangGraph 概念不熟，但有基本 Python/LLM 知识。  
> - 解释要**分层**：先画 high-level 流程，再落到关键文件/函数。  
> - 善用「文本框图」和项目内路径，例如：  
>   - `src/open_deep_research/deep_researcher.py:clarify_with_user`  
>   - `StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)`  
> - 关键名词出现时尽量说明 3 件事：是什么、在 LangGraph 官方文档里属于哪一类、在本项目里具体在哪里用。

> 技术细节要求  
> - 讲解时要指出：  
>   - 哪些地方在用 LangGraph 图结构（StateGraph、子图）；  
>   - 哪些地方在用 LangGraph 的状态与 reducer（MessagesState、Annotated[...]）；  
>   - 哪些地方在用工具和 MCP（`get_all_tools`、Tavily、MCP 客户端）；  
>   - 哪些地方在用 checkpointer / store（MemorySaver、get_store）。  
> - 避免泛泛而谈，只说“用了 LangGraph”；要具体到“在哪个文件的哪段代码用了哪个类/函数，作用是什么”。

> 本地环境相关（仅在需要时提及）  
> - 默认假定模型通过 OpenAI 兼容协议接在本地 vLLM，地址由 `.env` 中的 `OPENAI_BASE_URL` 和 dummy `OPENAI_API_KEY` 提供。  
> - 如果回答涉及运行示例，优先参考：`examples/simple_langgraph_agent.py` 和 README 中的 `langgraph dev` 命令。

在解释项目时，你可以把上面的内容当作“系统提示词”，用来约束自己的角色和输出风格。




1. 为什么是 async def
                                                                                                                       
  async def router(state: SimpleState, config: RunnableConfig) -> Command[...]:                                        
                                                                                                                       
  - 在 LangGraph 里，节点函数可以是同步的 def，也可以是异步的 async def。                                              
  - 本项目里所有节点（包括会调用模型的 answer、clarify_with_user 等）都是异步的，用 await model.ainvoke(...)。为了保持 
    风格一致，router 也写成 async。                                                                                    
  - 这样做的好处：                                                                                                     
      - 以后如果你想在 router 里加异步操作（比如查数据库、调 HTTP），不用再改函数签名。                                
      - LangGraph 对 async 节点支持得很好，整个执行链用 asyncio 并发。                                                 
                                                                                                                       
  所以：虽然这个函数里面暂时没有 await，写成 async 主要是为了与其他节点统一和可扩展。                                  
                                                                                                                       
  ———                                                                                                                  
                                                                                                                       
  2. state 和 config 是什么                                                                                            
                                                                                                                       
  state: SimpleState, config: RunnableConfig                                                                           
                                                                                                                       
  - state：                                                                                                            
      - 类型是 SimpleState，它继承自 MessagesState，所以至少有一个字段：messages。                                     
      - 在函数里你看到：state["messages"][-1]，就是从当前图的状态里取最后一条消息。                                    
      - 这就是 LangGraph 的核心设计：所有节点共享同一个 state，节点只能通过 state 交流信息。                           
  - config：                                                                                                           
      - 类型是 RunnableConfig（来自 langchain_core），是 LangGraph/LangChain 用来传“运行配置”的结构。                  
      - 里面常见字段：config["configurable"]（你自己的可配置参数）、config["metadata"]（例如 owner、thread_id）等。    
      - 在这个 router 里没用到 config，但保留参数方便以后扩展（比如根据配置选择不同路由）。                            
                                                                                                                       
  ———                                                                                                                  
                                                                                                                       
  3. -> Command[Literal["answer", "__end__"]] 是返回类型                                                               
                                                                                                                       
  ) -> Command[Literal["answer", "__end__"]]:                                                                          
                                                                                                                       
  - 这是 Python 的类型注解，告诉你和编辑器：                                                                           
        报错。
  ———

  简单总结：

  - async：为了兼容异步调用和和其他节点保持一致。
  - state / config：是 LangGraph 约定的节点参数，一个是共享状态，一个是运行配置。
  - -> Command[...]：是返回类型注解，说明函数要返回一个 Command，并限制可跳转的节点名。
  - 三引号的英文注释：是 Python 函数 docstring，起说明和 IDE 提示作用。

  如果你想，我可以把这个 router 改成更复杂一点的版本，比如根据问题内容选择走“聊天节点”还是“搜索节点”，帮你感受一下     
  LangGraph 里的“路由节点”是怎么写的。