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




### 1. 为什么是 async def
                                                                                                                       
  async def router(state: SimpleState, config: RunnableConfig) -> Command[...]:                                        
                                                                                                                       
  - 在 LangGraph 里，节点函数可以是同步的 def，也可以是异步的 async def。                                              
  - 本项目里所有节点（包括会调用模型的 answer、clarify_with_user 等）都是异步的，用 await model.ainvoke(...)。为了保持 
    风格一致，router 也写成 async。                                                                                    
  - 这样做的好处：                                                                                                     
      - 以后如果你想在 router 里加异步操作（比如查数据库、调 HTTP），不用再改函数签名。                                
      - LangGraph 对 async 节点支持得很好，整个执行链用 asyncio 并发。                                                 
                                                                                                                       
  所以：虽然这个函数里面暂时没有 await，写成 async 主要是为了与其他节点统一和可扩展。                                  
                                                                                                                       
  ———                                                                                                                  
                                                                                                                       
###   2. state 和 config 是什么                                                                                            
                                                                                                                       
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
                                                                                                                       
###  3. -> Command[Literal["answer", "__end__"]] 是返回类型                                                               
                                                                                                                       
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




### • dataclass 不是一种“独立语法”。它本质上是 标准库 dataclasses 里的一个装饰器函数：@dataclass。
                                                                                                        
  它做的事是：在你定义完普通 class 之后，自动给这个类补上/生成一些方法（比如 __init__、__repr__、__eq__ 
  等），以及根据字段定义做一些额外处理。                                                                
                                                                                                        
  所以更准确的说法是：                                                                                  
                                                                                                        
  - @dataclass：装饰器（函数），用在类定义上                                                            
  - 被装饰的东西：仍然是一个普通的 Python 类                                                            
  - “像结构体”的感觉：是因为它让类更像“纯数据容器”，但 Python 没有内置 struct 类型（那个是 C/Rust 的    
    概念）    



### › Command[Literal["confirm", "__end__"]]: 这里返回的数据结构是什么样的

• Command[Literal["confirm", "__end__"]] 里的 Command[...] 是一个 Python dataclass 对象（不是 dict），真
  正的数据结构就是 langgraph.types.Command，里面有 4 个字段（你可以理解成“给 LangGraph 引擎的指令包”）：

  - goto：下一步要跳转到哪个节点（这里限制只能是 "confirm" 或 "__end__"；"__end__" 就是 END）
  - graph：把命令发给哪个子图（一般用不到，默认 None）

  你在 simple_langgraph_agent.py:58 里写的 return Command(goto="confirm")，实际对象长这样（等价于）：   

  - 打印出来：Command(goto='confirm')
  - 转成 dict（便于理解）：{'graph': None, 'update': None, 'resume': None, 'goto': 'confirm'}

  Command[Literal["confirm", "__end__"]] 这个 [...] 只是类型标注（给 IDE/类型检查看的），运行时返回的还 
  是同一个 Command 对象。



### Python dataclass 对象介绍一下概念
• dataclass 是 Python 里用来“快速定义数据结构（数据容器）”的一种语法糖（模块名：dataclasses，Python     
  3.7+）。

  核心概念

  - 你写一个类，主要目的是“装数据”（一堆字段），而不是复杂行为。
  - 加上 @dataclass 后，Python 会自动帮你生成常用的样板代码：
      - __init__：构造函数，自动接收字段参数
      - __repr__：打印时好看（比如 User(name='a', age=1)）
      - __eq__：可以按字段比较是否相等
      - （可选）__hash__、排序方法等

  最小例子

  from dataclasses import dataclass

  @dataclass
  class User:
      name: str
      age: int

  u = User(name="Alice", age=18)
  print(u)          # User(name='Alice', age=18)
  print(u.name)     # Alice

  常见可选参数

  - frozen=True：实例变成“只读”（像不可变对象）
  - kw_only=True：强制只能用关键字传参（更不容易写错位置参数）
  - slots=True：更省内存、属性更快（但更严格）

  和普通 class 的区别（直觉）

  - 普通 class：你要自己写 __init__、__repr__ 等。
  - @dataclass：你声明“有哪些字段”，其余样板代码自动生成。


› - __init__：构造函数，自动接收字段参数
        - __repr__：打印时好看（比如 User(name='a', age=1)）
        - __eq__：可以按字段比较是否相等
        - （可选）__hash__、排序方法等        这些都是普通类的构造函数对





###  › - __init__：构造函数，自动接收字段参数
        - __repr__：打印时好看（比如 User(name='a', age=1)）
        - __eq__：可以按字段比较是否相等
        - （可选）__hash__、排序方法等        这些都是普通类的构造函数对


• 这些都叫“魔术方法/双下划线方法（dunder methods）”，普通类当然也能有，只是你通常要自己写；@dataclass 的
  价值是“按字段自动帮你生成”。

  - __init__：初始化方法（很多人口头叫构造函数，但严格说 Python 的对象创建是 __new__，__init__ 负责“创建
    __repr__(self): ...。
  - __eq__：决定 a == b 怎么比较。普通类也能手写 def __eq__(self, other): ...。
  - __hash__：决定对象能不能作为 dict/set 的 key，以及哈希值是什么；它和 __eq__ 有联动（实现了按值相等  
    后，默认通常会让对象不可哈希，避免“相等但哈希不同”的坑）。普通类也能手写。
  所以：不是 dataclass 专属；dataclass 只是根据你声明的字段，帮你把这些“模板代码”自动生成出来。
          return super().__new__(cls)

      def __init__(self, name: str, age: int, tags=None):
              raise ValueError("name must be a non-empty str")
          if not isinstance(age, int) or age < 0:
              raise ValueError("age must be a non-negative int")
          self.name = name

      def __repr__(self) -> str:
          # 交互式/print 时的“官方表示”
          return f"User(name={self.name!r}, age={self.age!r}, tags={self.tags!r})"

      def __eq__(self, other) -> bool:
          # 按字段比较相等
          if not isinstance(other, User):
              return NotImplemented
          return (self.name, self.age, self.tags) == (other.name, other.age, other.tags)

      def __hash__(self) -> int:
          # 让它能当 dict/set 的 key（注意：tags 是 list，不可哈希，需要转 tuple）
          return hash((self.name, self.age, tuple(self.tags)))

      def __lt__(self, other) -> bool:
          # 排序规则：先按 age，再按 name
          if not isinstance(other, User):
              return NotImplemented
          return (self.age, self.name) < (other.age, other.name)


  u1 = User("Alice", 18, tags=["vip"])
  u2 = User("Alice", 18, tags=["vip"])
  u3 = User("Bob", 20)

  print(u1)                 # User(name='Alice', age=18, tags=['vip'])
  print(u1 == u2)           # True
  print({u1: "ok"}[u2])     # ok（因为 hash/eq 一致）
  print(sorted([u3, u1]))   # 按 __lt__ 排序
› u1 = User("Alice", 18, tags=["vip"])
    u2 = User("Alice", 18, tags=["vip"])
    u3 = User("Bob", 20)这里在执行时候，使用到了def的哪个方法




### __eq__ 只有在 u1 == u2 时才会被调用。  两个类相等的判定条件是什么


• 在我们刚才那个 User 例子里，“两个对象相等”的条件是你在 __eq__ 里自己写死的。

  例子回顾：

  class User:
      ...
      def __eq__(self, other) -> bool:
          # 按字段比较相等
          if not isinstance(other, User):
              return NotImplemented
          return (self.name, self.age, self.tags) == (other.name, other.age, other.tags)

  所以对于 u1 == u2：

  - 先看 other 是否是 User 实例（isinstance(other, User)）
  - 然后比较三个字段完全相同：
      - self.name == other.name
      - self.age == other.age
      - self.tags == other.tags（这里是列表内容相同）

  只有这三项都相等时，__eq__ 返回 True，于是 u1 == u2 才算“相等”。

  更一般地说：

  - Python 里 a == b 的判定规则就是：调用 a.__eq__(b)（再加上一些回退逻辑），你在 __eq__ 里写什么，就用 
    什么标准判定相等。
  - 如果你不自己写 __eq__，默认继承自 object.__eq__，那就是“同一个对象才相等”（id(a) == id(b)）。


### 

