---
title: "LangChain / LangGraph 系统学习总结 —— 基于 langchain-examples 仓库"
lang: zh-CN
---

# LangChain / LangGraph 系统学习总结

> 本总结以你的仓库 `langchain-examples`（`C:\repos\github.com\langchain-examples`）中的 **11 个源文件、约 950 行代码** 为主要学习材料，逐一分析每个文件、每个关键 API，并在此基础上补齐官方概念体系，直到你能**不依赖这份仓库、自己从零设计一个 LangChain / LangGraph Agent 应用**为止。
>
> 标注约定：✅ = 仓库代码中实际出现　📘 = 补充官方知识（仓库未出现，但是理解全貌的前置/后续内容）　⚠️ = Legacy / Deprecated 写法　🔑 = Prerequisite（后续内容的前提）　⭐ = Advanced　🏭 = Production 中特别重要。

---

## 目录

1. [LangChain / LangGraph 是什么](#1)
2. [LangChain 核心概念：Model](#2)
3. [LangChain 核心概念：Prompt](#3)
4. [Runnable 与 LCEL（`|` 到底是什么）](#4)
5. [Output Parser / Structured Output](#5)
6. [Tool](#6)
7. [Agent](#7)
8. [Memory / Conversation State](#8)
9. [RAG —— 本仓库的核心内容](#9)
10. [LangGraph 是什么](#10)
11. [LangGraph 核心概念：State / Node / Edge](#11)
12. [LangGraph Agent 完整示例](#12)
13. [LangChain vs LangGraph](#13)
14. [从简单到复杂的 5 个项目](#14)
15. [Sample Code 逐个分析](#15)
16. [知识体系重组](#16)
17. [常见误区（15 条）](#17)
18. [Debugging / Troubleshooting](#18)
19. [Production Considerations](#19)
20. [学习路线（Level 1–6）](#20)
21. [Cheat Sheet](#21)
22. [Version / API 变化说明](#22)
23. [自测题](#23)

---

## 仓库覆盖情况地图

在开始之前，先说清楚这份仓库实际"educates"你哪些内容，避免你以为代码里出现过其实并没有的东西。

| 主题 | 仓库是否有代码 | 对应文件 |
|---|---|---|
| Document 加载 | ✅ 有 | `documents.py` |
| Text Splitter | ✅ 有 | `text_splitter.py` |
| Embeddings | ✅ 有 | `embeddings.py` |
| Vector Store（Chroma） | ✅ 有 | `embeddings_chroma.py` |
| Retriever（含 MMR） | ✅ 有 | `embedding_chroma_retrieval.py` |
| 向量库持久化 | ✅ 有 | `embedding_chroma_persistence.py` |
| Chat Model 调用 | ✅ 有（无 Prompt Template） | `check_connection.py` |
| 配置管理模式 | ✅ 有 | `config.py` |
| RAG 生成端（Prompt+LLM 拼接检索结果） | ⚠️ 只有注释掉的 stub | `rag_pipeline.py` |
| Prompt Template / LCEL `\|` / Output Parser | ❌ 没有 | — |
| Tool / Tool Calling | ❌ 没有 | — |
| Agent（LangChain 或 LangGraph） | ❌ 没有 | — |
| Memory / Message History | ❌ 没有 | — |
| Structured Output（Pydantic） | ❌ 没有 | — |
| Streaming / Async | ❌ 没有 | — |
| LangGraph（State/Node/Edge） | ❌ 没有 | — |

一句话结论：**这个仓库把 RAG 的"检索半段"（Ingestion → Split → Embed → Store → Retrieve）做得很扎实，但"生成半段"（Prompt 拼接 + LLM 生成答案）以及 Tool / Agent / LangGraph 完全没有落地**。本总结第 9 节会把已有代码讲透，第 2–8、10–13 节会用官方概念把缺的部分补齐，并且在第 14 节给你 5 个从易到难、直接可以在这个仓库基础上练手的项目，其中就包括把 `rag_pipeline.py` 补完。

---

<a id="1"></a>
## 1. LangChain / LangGraph 是什么

### 1.1 什么是 LLM Application 🔑 Prerequisite

一个"LLM 应用"不是"调一下 API、把回答打印出来"这么简单。真实的应用通常要把下面几个部件组合起来：

- **LLM / Chat Model**：接收文本（或消息列表），返回文本（或消息）。本质上是一个"输入字符串 → 输出字符串"的函数，只是这个函数极其强大、也极不确定（同样输入可能得到不同输出）。
- **Prompt**：喂给模型的指令模板。把"系统角色设定 + 用户输入 + 少量示例"拼接成模型能理解的格式，是控制模型行为的第一道杠杆。
- **Tool**：模型本身不能查数据库、发邮件、算数学——Tool 是把这些"外部能力"包装成模型可以"请求调用"的函数。
- **Agent**：让模型自己决定"要不要调用 Tool、调用哪个、调用之后下一步做什么"的循环控制逻辑。
- **Memory**：让多轮对话之间能记住上下文，而不是每次都从零开始。
- **RAG（Retrieval-Augmented Generation）**：模型的知识停留在训练时间点，RAG 通过"先检索相关资料，再把资料塞进 Prompt"来让模型回答它本来不知道的、或者更新的内容——你的仓库做的正是这件事的前半段。
- **Workflow / Graph**：当一个任务需要多个步骤、有分支、甚至要循环重试时，把这些步骤显式地编排成一张图（这就是 LangGraph 要解决的问题）。

它们的关系可以理解成一个逐层递进的"能力叠加"：

```text
LLM                         —— 只能对话，什么都不知道，什么都做不了
LLM + Prompt                —— 能被"教"成特定角色/特定任务
LLM + Prompt + Tool         —— 能查资料、能算数、能调 API
LLM + Prompt + Tool + Loop  —— 这就是 Agent：自己决定下一步
Agent + Memory              —— 记得住之前说过什么
Agent + RAG                 —— 能基于你自己的私有数据回答
Agent + Graph（LangGraph）   —— 复杂、可控、可持久化、可人工介入的工作流
```

整体架构图：

```text
User
  │
  ▼
Application（你写的业务代码）
  │
  ▼
LangChain（组件与标准接口）/ LangGraph（编排与状态机）
  │
  ▼
LLM（Chat Model，例如 Bedrock 上的 Nova / Claude）
  │
  ├──▶ Tools（函数、API 调用）
  ├──▶ Database / 业务系统
  └──▶ Vector Store（Chroma 等，用于 RAG）
```

### 1.2 LangChain 解决什么问题

如果没有 LangChain，直接调用模型厂商的 SDK 也能写出一个应用，但会遇到三个反复出现的痛点：

1. **每换一个模型供应商，代码就要重写**——LangChain 用统一的 `ChatModel` / `Runnable` 接口屏蔽了这些差异（你的仓库里就体现为：如果哪天从 AWS Bedrock 换成 OpenAI，理论上只需要换 `ChatBedrockConverse` 为另一个 Chat Model 类，`check_connection.py` 里 `llm.invoke([...])` 这行代码几乎不用动）。
2. **"prompt → model → 解析结果 → 再喂给下一步"这种链式组合，手写胶水代码又臭又长**——LangChain 用 LCEL（`|` 操作符）把这种组合标准化。
3. **检索、分块、Embedding、向量库这些 RAG 的基础设施，每个项目都要重新写一遍**——LangChain 把它们抽象成统一接口（`Document`、`TextSplitter`、`Embeddings`、`VectorStore`、`Retriever`），你的仓库正是在用这套接口。

### 1.3 LangGraph 解决什么问题（先给一个预告，第 10 节详细展开）

LangChain 的 Agent 本质上是一个"模型说了算"的循环：模型自己决定调不调 Tool、调几次、什么时候停。这在简单场景下够用，但当你需要——

- 明确规定"步骤 A 一定要在步骤 B 之前"（而不是让模型自由发挥）；
- 引入循环、重试、条件分支；
- 长时间运行的任务需要中途保存状态、下次接着跑；
- 关键操作前必须停下来等人审批（Human-in-the-loop）；
- 多个 Agent 协作、互相传递状态；

——这些"确定性的流程控制 + 显式状态"就是 LangChain 的 Agent 循环不擅长、而 LangGraph 专门解决的问题。LangGraph **不是取代 LangChain**，而是建立在 LangChain 的 Model / Tool / Message 之上，多加了一层"图"。

---

<a id="2"></a>
## 2. LangChain 核心概念：Model

### 2.1 LLM vs Chat Model

早期 LangChain 区分 `LLM`（纯文本补全，输入字符串、输出字符串）和 `ChatModel`（输入/输出都是"消息列表"，有 role 概念：system / human / ai）。**现在几乎所有主流模型都是 Chat 接口**，纯 `LLM` 类已经是历史遗留概念，新代码应该总是用 Chat Model。

### 2.2 ✅ 仓库中的体现：`ChatBedrockConverse`

`check_connection.py` 是仓库里唯一直接调用 Chat Model 的地方：

```python
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

llm = ChatBedrockConverse(
    model=model_id,
    region_name=config.aws.region,
    aws_access_key_id=config.aws.access_key_id,
    aws_secret_access_key=config.aws.secret_access_key,
    aws_session_token=config.aws.session_token,
)

response = llm.invoke([HumanMessage(content="What is the capital city of China")])
print(response.content)
```

逐行拆解：

- `ChatBedrockConverse` 是 `langchain_aws` 提供的 Chat Model 实现，底层用的是 AWS Bedrock 的 **Converse API**——这是关键设计：Bedrock 上有 Amazon Nova、Anthropic Claude、Meta Llama 等多家模型，每家原始请求格式都不同，Converse API 把它们统一成一种请求/响应形状，`ChatBedrockConverse` 再把这套统一接口包装成 LangChain 的 `BaseChatModel`。这是**"为什么需要 Chat Model 抽象"**的一个非常好的真实例子：即使你只用 Bedrock，模型换来换去也不用改调用代码。
- `HumanMessage(content=...)` 是"消息"而不是裸字符串——这是 Chat Model 和旧式 LLM 接口的本质区别：Chat Model 的输入是一个**消息列表**（`list[BaseMessage]`），每条消息带 role。仓库这里只用了最简单的一条 `HumanMessage`，但真实应用通常是 `[SystemMessage(...), HumanMessage(...), AIMessage(...), HumanMessage(...)]` 这样的多轮历史。
- `.invoke(messages)` 是**标准 Runnable 接口**（第 4 节详细讲）——`ChatBedrockConverse` 本质上是一个 `Runnable[list[BaseMessage], AIMessage]`，`.invoke()` 是所有 Runnable 通用的调用方法，这也是为什么它能被无缝塞进 LCEL 链条里的 `prompt | model` 组合。
- `response.content` 是返回的 `AIMessage` 对象的文本内容；`AIMessage` 除了 `content` 还有 `tool_calls`、`usage_metadata`（token 用量）等字段，这在后面 Tool Calling（第 6 节）时会用到。
- `region_name` / `aws_access_key_id` 等参数直接来自 `config.py` 的 `Config` 对象——**这是一个值得学的工程实践**：把"从哪读凭证"和"怎么用凭证建模型客户端"分离开，模型构建代码只依赖一个类型化的 `Config`，不直接读环境变量。

### 2.3 Message 类型 🔑

`langchain_core.messages` 里的几个核心类：

| 类 | role | 仓库是否用到 | 用途 |
|---|---|---|---|
| `SystemMessage` | system | ❌ 未用 | 设定角色/行为约束，通常是对话第一条 |
| `HumanMessage` | user | ✅ `check_connection.py` | 用户输入 |
| `AIMessage` | assistant | ✅（作为返回值） | 模型的回复，可能带 `tool_calls` |
| `ToolMessage` | tool | ❌ 未用 | Tool 执行结果回传给模型（第 6 节） |

仓库里**没有出现 `SystemMessage`**，也就是说 `check_connection.py` 里的模型是"裸调用"，没有任何角色设定——这对一个纯粹验证连通性的脚本没问题，但真实应用几乎总会带一条 `SystemMessage` 来控制模型的行为边界，这也是下一节 Prompt Template 要解决的问题。

### 2.4 Temperature / Model Configuration 📘

仓库的 `ChatBedrockConverse` 构造时没有传 `temperature`，用的是模型默认值。补充概念：`temperature` 控制输出的随机性（0 = 几乎确定性输出，1+ = 更发散），`max_tokens` 控制输出长度上限。生产环境中这些通常也应该做成可配置项，走 `config.py` 这种模式而不是硬编码。

### Key Takeaway

Chat Model 是"输入消息列表、输出一条 AI 消息"的标准化组件；它是 `Runnable`，因此天生可以用 `.invoke()` 调用，也天生可以被塞进 LCEL 链条。仓库的 `check_connection.py` 展示了最简形态；下一步自然的补全是加上 `SystemMessage`（角色设定）和 `PromptTemplate`（变量化的提示词）——这正是第 3 节要讲的内容。

---

<a id="3"></a>
## 3. LangChain 核心概念：Prompt

### 📘 本仓库未覆盖 —— 补充知识

仓库里没有任何 `PromptTemplate` 或 `ChatPromptTemplate` 的使用，`check_connection.py` 是把字符串硬编码进 `HumanMessage(content="...")`。这里补充为什么真实项目几乎从不这样写。

### 3.1 为什么不能只用字符串拼接

假设你要写"请解释 {topic}"这种带变量的提示词，最朴素的写法是 f-string：

```python
prompt_text = f"You are a helpful assistant. Explain {topic}."
```

这在变量少、逻辑简单时能用，但很快会遇到问题：多轮消息（system + human）怎么拼？变量需要类型校验和默认值怎么办？想把同一个 Prompt 在不同链里复用怎么办？`ChatPromptTemplate` 就是为了系统性解决这些问题：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic}")
])

messages = prompt.invoke({"topic": "LangChain"})
# messages 是一个 ChatPromptValue，可以 .to_messages() 得到
# [SystemMessage(...), HumanMessage(content="Explain LangChain")]
```

- `ChatPromptTemplate` 本身也是一个 **Runnable**（`Runnable[dict, ChatPromptValue]`）——它接收一个字典（变量），输出可以直接喂给 Chat Model 的消息列表。这就是为什么 `prompt | model` 能直接用 `|` 连起来：两者都实现了同一套 Runnable 接口。
- `{topic}` 是模板变量，`.invoke({"topic": "LangChain"})` 时做替换，如果变量缺失会在这一步直接报错（而不是等到模型调用失败），这是比手写 f-string 更早暴露问题的好处。

### 3.2 Prompt Template 与普通字符串的区别

| | f-string / 手写字符串 | `ChatPromptTemplate` |
|---|---|---|
| 多角色消息（system/human/ai） | 需要自己拼装成列表 | 原生支持 |
| 变量校验 | 运行时才会因为 KeyError 报错 | `.invoke()` 时统一校验 |
| 复用/组合 | 复制粘贴 | 可以和其他 Runnable 用 `\|` 组合 |
| 与 Few-shot 示例结合 | 手写 | `FewShotPromptTemplate` 原生支持 |
| 序列化/落盘 | 不方便 | 支持保存/加载模板 |

### 3.3 如果给仓库补一个 Prompt Template（示范代码，非仓库现有代码）

结合 `check_connection.py` 现有的裸调用，一个更贴近真实项目的版本：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant. Answer in one sentence."),
    ("human", "{question}"),
])

chain = prompt | llm   # llm 就是仓库里已经建好的 ChatBedrockConverse 实例
response = chain.invoke({"question": "What is the capital city of China"})
print(response.content)
```

这一行 `prompt | llm` 就是第 4 节要讲的 LCEL。

### Key Takeaway

Prompt Template 把"提示词"从散落在业务代码里的字符串，提升为一个可复用、可校验、可与其他组件组合的 Runnable 组件。仓库目前处在"连 Prompt Template 都还没引入"的阶段，这是最值得补的第一课。

---

<a id="4"></a>
## 4. Runnable 与 LCEL（`|` 到底是什么）

### 📘 本仓库未直接出现，但有"伏笔"

仓库虽然没写出 `chain = prompt | model` 这种代码，但 `embedding_chroma_retrieval.py` 里的这段注释其实已经在解释 Runnable 概念了：

```python
# Retrievers are Runnables, so .invoke(query) is the standard way to
# call one - the same interface a prompt or chat model uses, which is
# what lets a retriever slot into an LCEL chain.
results: list[Document] = retriever.invoke(query)
```

也就是说：**仓库里的 `Retriever`、`ChatBedrockConverse`、`BedrockEmbeddings` 全部都是 `Runnable`**，只是仓库从未把它们用 `|` 连接起来用过，都是逐个手动调用（先 `similarity_search`，再打印，再下一个函数）。这一节把这层被"隐藏"的抽象讲清楚。

### 4.1 Runnable 是什么 🔑

`Runnable` 是 LangChain 定义的一个**协议（接口）**：任何实现了 `invoke` / `batch` / `stream`（以及对应的异步版本 `ainvoke` / `abatch` / `astream`）方法的对象，都是一个 Runnable。Chat Model、Prompt Template、Output Parser、Retriever、甚至一个普通 Python 函数（包一层 `RunnableLambda`）——全部统一在这一个接口下。

```text
Runnable
 ├─ .invoke(input) -> output          单次调用
 ├─ .batch([input1, input2, ...])     批量调用
 ├─ .stream(input)                    流式输出（逐 token/逐块）
 └─ .ainvoke / .abatch / .astream     以上三个的异步版本
```

这解决的核心问题是：**只要两个组件都符合这个接口，就可以互相拼接，不用关心内部实现**。这正是为什么仓库里 `vector_store.as_retriever()` 返回的对象可以直接 `.invoke(query)` 调用，用法和调用 `llm.invoke(messages)` 长得一模一样。

### 4.2 `|` 是什么

```python
chain = prompt | model | parser
```

`|` 是 Python 的运算符重载——`Runnable` 类实现了 `__or__` 方法，`a | b` 实际上构造了一个 `RunnableSequence(a, b)`。调用 `chain.invoke(x)` 时，等价于：

```python
parser.invoke(model.invoke(prompt.invoke(x)))
```

即"前一个的输出，自动作为后一个的输入"，你不需要手写这层胶水代码。数据流向：

```text
Input（dict，例如 {"topic": "LangChain"}）
  │
  ▼
Prompt（把变量填进模板，输出消息列表）
  │
  ▼
Model（把消息列表喂给 LLM，输出 AIMessage）
  │
  ▼
Output Parser（把 AIMessage 转成你要的格式，比如纯字符串或 JSON）
  │
  ▼
Output
```

### 4.3 常用 Runnable 组合器

| 类 | 作用 | 类比 |
|---|---|---|
| `RunnableSequence`（即 `\|`） | 顺序执行，前一个输出是后一个输入 | 流水线 |
| `RunnableParallel` | 同时对同一个输入跑多个 Runnable，结果汇总成 dict | 并发扇出 |
| `RunnableLambda` | 把一个普通 Python 函数包装成 Runnable | 适配器 |
| `RunnablePassthrough` | 原样传递输入（常用于在 `RunnableParallel` 里保留原始输入） | 直通线 |

`RunnableParallel` 示例（补充代码，仓库中未出现）：

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 典型 RAG 场景：一边把原始 question 透传下去，一边用它去检索
chain = RunnableParallel(
    context=retriever,              # 复用仓库里已有的 retriever
    question=RunnablePassthrough(),
)
```

### 4.4 为什么大量使用这种 composable architecture

因为 LLM 应用的本质就是"多个变换步骤串联/并联"，用统一接口组合，换掉任何一环（换个模型、换个向量库、加一步后处理）都不需要改动其它环节的代码——这和仓库 `CLAUDE.md` 里强调的"不依赖 `langchain-community`、直接包装底层库"是同一种工程哲学的两个体现：**都是在追求"每一层只做一件事、层与层之间用稳定接口解耦"**。

### Common Mistakes（通用）

- 把 `.invoke()` 的返回值类型搞混——`prompt.invoke()` 返回 `ChatPromptValue`，`model.invoke()` 返回 `AIMessage`，直接把 `AIMessage` 当字符串用会出错（要用 `.content`，或者接一个 `StrOutputParser`）。
- 在 `RunnableParallel` 里忘记用 `RunnablePassthrough()` 保留原始输入，导致后续步骤拿不到原始问题。

### Key Takeaway

Runnable 是"万物皆可 `.invoke()`"的统一协议；`|` 只是这个协议之上的语法糖。理解了这一点，你会发现仓库里的 `retriever.invoke(query)` 和 `llm.invoke(messages)` 并不是两套不相关的 API，而是同一套接口的两个实例——这也是为什么把它们用 `|` 拼在一起（正是 `rag_pipeline.py` 应该做但还没做的事）会如此自然。

---

<a id="5"></a>
## 5. Output Parser / Structured Output

### 📘 本仓库未覆盖 —— 补充知识

`check_connection.py` 直接打印 `response.content`（一段自由文本）。仓库里从未要求模型返回结构化数据。

### 5.1 为什么需要 Output Parser

模型默认输出的是**自由文本**。如果你的下游代码要把结果存进数据库字段、做逻辑判断，光有一段文字是不够的，你需要模型"稳定地"按某种格式输出（比如 JSON），并且解析成 Python 对象。

### 5.2 为什么不能完全依赖 LLM 自己返回 JSON

即便你在 Prompt 里写"请用 JSON 格式回答"，模型仍然可能：多打一句解释、漏个引号、字段名拼错、数值精度不对。**Structured Output 不是"祈祷模型格式正确"，而是通过工具调用（tool calling）机制，让模型的输出被强制约束到一个 Schema**，这比单纯指望模型"听话"可靠得多。

### 5.3 `with_structured_output` + Pydantic

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

structured_model = model.with_structured_output(Person)
result = structured_model.invoke("John is 30 years old.")
# result 是一个 Person(name="John", age=30) 实例，不是字符串
```

- `Person` 是一个 **Pydantic Schema**，描述"我要的数据长什么样"。
- `with_structured_output(Person)` 返回一个新的 Runnable，内部会把 `Person` 的字段定义转换成模型能理解的 Tool/Schema 定义，底层通常借助模型的 tool-calling 能力（第 6 节）来强制约束输出形状，再由 Pydantic 做一次校验（类型不对会直接抛 `ValidationError`）。
- `.invoke()` 的返回值不再是 `AIMessage`，而是**校验通过的 `Person` 对象**——这是 Structured Output 相对普通 Output Parser（比如 `JsonOutputParser`）更强的地方：Pydantic 会做类型转换和校验，而不只是"解析成一个 dict"。

### Common Mistakes

- Schema 定义得太复杂（嵌套过深、字段过多），模型出错率会上升——Schema 应该尽量扁平、字段语义清晰。
- 忘记模型本身要支持 tool calling——不是所有模型/所有版本都支持 `with_structured_output`。

### Key Takeaway

Structured Output 把"LLM 输出"从"一段可能不稳定的文本"，变成"一个有类型保证的 Python 对象"，是把 LLM 结果接入正常业务逻辑（而不只是展示给人看）的关键一步。

---

<a id="6"></a>
## 6. Tool

### 📘 本仓库未覆盖 —— 补充知识

### 6.1 什么是 Tool，为什么 Agent 需要它

LLM 本身只会"生成文本"，它不能查数据库、发 HTTP 请求、做精确计算（模型做算术其实是"猜"，不可靠）。**Tool 就是把这些外部能力包装成一个模型可以"请求调用"的函数**，模型自己不执行 Tool，它只是"决定要调用哪个 Tool、传什么参数"，真正的执行是你的代码在做。

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

- `@tool` 装饰器做了两件事：（1）把函数包装成一个 `BaseTool` 对象（同样是 Runnable）；（2）**从函数签名和 docstring 自动生成一份 Tool Schema**（名字、参数类型、参数说明、返回类型），这份 Schema 会被发给模型，模型看到的不是 Python 代码，而是这份 Schema 描述。docstring 不是装饰性的注释——它直接决定模型"什么时候会想调用这个工具"，写得不清楚模型就调不准。

### 6.2 Tool Calling 的完整流程

```text
LLM
 │  看到用户问题 + 可用 Tool 的 Schema 列表
 ▼
决定要不要调用 Tool（如果不需要外部信息，直接回答）
 │
 ▼ （需要）
返回一个不含最终答案、但带 tool_calls 字段的 AIMessage
 │
 ▼
你的代码执行对应的 Tool（LangChain 不会替你自动执行！）
 │
 ▼
把 Tool 的返回值包装成 ToolMessage，追加回消息历史
 │
 ▼
再次调用 LLM，这次带着 Tool 的执行结果
 │
 ▼
LLM 生成最终答案（或者决定再调用一次别的 Tool）
```

关键点：**Tool 本身不会思考、不会"决定"任何事**——"要不要调用、调用哪个"完全是 LLM 根据 Schema 描述做出的判断（这也是第 17 节"常见误区"里会强调的一条：Tool 本身不进行 reasoning，reasoning 永远发生在 LLM 里）。

### Common Mistakes

- Tool 的 docstring 写得太模糊（比如只写 `"""Do something."""`），模型经常调错或者根本不调用。
- 忘记 Tool 执行结果需要包装成 `ToolMessage` 并且带上正确的 `tool_call_id`，否则模型对不上是哪次调用的结果。
- 让 Tool 直接抛异常而不是返回一个可读的错误字符串——模型看不到 Python Traceback，只能看到你传回去的文本。

### Key Takeaway

Tool = "函数 + 给模型看的 Schema"。它把 Agent 的能力从"只能说"扩展到"能做事"，但执行权始终在你的代码里，不在模型里——这也是为什么 Tool 权限控制（第 19 节）是生产环境的安全重点。

---

<a id="7"></a>
## 7. Agent

### 📘 本仓库未覆盖 —— 补充知识

### 7.1 Agent 到底是什么

对比一下 **Chain**（固定步骤）和 **Agent**（模型自己决定步骤）：

```text
Traditional Chain（步骤在写代码时就固定死了）
Input → Step 1 → Step 2 → Step 3 → Output

Agent（步骤由模型在运行时动态决定）
Input
  ▼
LLM ──▶ 决定下一步动作（Reasoning + Action）
  ▼
Tool ──▶ 执行，产生 Observation
  ▼
LLM ──▶ 结合 Observation，再决定下一步
  ▼
   ...（循环，直到 LLM 认为已经可以给出最终答案）
  ▼
Final Answer
```

Chain 是你替模型规划好了路线；Agent 是你只给模型一套工具箱和目标，路怎么走由模型自己摸索。

### 7.2 Agent 为什么会循环

因为很多任务不是"一次 Tool 调用就能搞定"的——模型可能需要先查一个信息，根据结果决定要不要再查别的，如此反复，直到收集到足够信息才能给出最终答案。这个"LLM → Tool → LLM → Tool → ... → LLM"的循环，就是 Agent Loop，也是 LangGraph 用一张"带环的图"（第 10 节）能够更显式地表达和控制的地方。

### 7.3 ⚠️ Legacy 与 ✅ Recommended（2026 现状）

LangChain 历史上有过好几代 Agent API：

| 时期 | API | 状态 |
|---|---|---|
| 早期 | `initialize_agent` | ⚠️ 已废弃 |
| 中期 | `AgentExecutor` + `create_react_agent`（`langchain.agents`） | ⚠️ Legacy，仍可用但不再是推荐路径 |
| 之后一段时间 | `langgraph.prebuilt.create_react_agent` | ⚠️ 已在 LangChain/LangGraph 1.0 中被标记为 deprecated |
| **现在（LangChain 1.0+）** | `langchain.agents.create_agent` | ✅ **推荐**，内部构建在 LangGraph 之上 |

`create_agent` 的最简用法：

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",   # 或直接传一个已构造好的 Chat Model 实例
    tools=[search],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

官方把这个心智模型总结为一句话：**Agent = Model + Harness**（Harness 指“模型循环 + 状态管理 + Tool 执行”这套周边基础设施）。`create_agent` 内部就是用 LangGraph 构建了一个持久化状态图，`messages` 是这个图状态里的一个字段——这直接说明了第 13 节要讲的"LangChain 和 LangGraph 不是替代关系"：**新版 LangChain 的 Agent 本身就是一个预先搭好的 LangGraph 图**，`create_agent` 是"帮你把图搭好"的高层封装；需要更细粒度控制（自定义节点、条件分支、人工审批）时，直接下沉到 LangGraph 的 `StateGraph` API（第 10–12 节）。

### Key Takeaway

Agent 的本质是"LLM 在一个 Tool 集合上做循环决策"。理解 Agent，本质上就是理解"Reasoning（模型思考下一步）→ Action（调用 Tool）→ Observation（拿到结果）"这个三段循环——RAG 之后如果仓库要进化成一个"能自己决定要不要检索、要检索几次"的系统，走的就是这条路。

---

<a id="8"></a>
## 8. Memory / Conversation State

### 📘 本仓库未覆盖 —— 补充知识

### 8.1 为什么模型"记不住"上一轮说了什么

Chat Model 本身是**无状态**的——`llm.invoke(messages)` 每次调用都是独立的一次请求，模型看到的只有你这一次传进去的 `messages` 列表。所谓"记住上文"，其实是**调用方在每次请求时，把完整的历史消息列表重新传进去**：

```text
User: My name is John.
  → 发送 [HumanMessage("My name is John.")]
  ← AIMessage("Nice to meet you.")

User: What's my name?
  → 发送 [HumanMessage("My name is John."), AIMessage("Nice to meet you."), HumanMessage("What's my name?")]
  ← AIMessage("Your name is John.")
```

第二次请求之所以能回答对，纯粹是因为**你把第一轮的完整对话历史又发了一遍**，模型本身没有任何"记忆存储"。

### 8.2 LangChain 中如何管理 Conversation History

- 最基础：自己维护一个 `list[BaseMessage]`，每轮往后追加。
- `RunnableWithMessageHistory`（较早期方案）：给一个 Runnable 包一层，自动按 `session_id` 存取历史。
- **在 LangGraph 里**：对话历史通常直接作为 `State` 的一个字段（例如 `messages: list[BaseMessage]`），配合 `checkpointer`（第 11 节）持久化，这是目前更推荐的做法——第 8 节和第 10 节在这里正式产生交集。

### 8.3 Memory 和 State 的区别 🔑

这是一个非常容易混淆、也是第 17 节"常见误区"专门会点名的点：

| | Memory | State |
|---|---|---|
| 关注点 | "让模型记得对话上下文" | "整个应用当前所处的完整数据快照" |
| 典型内容 | 消息历史（有时会做摘要/裁剪） | 消息历史 + 业务变量（比如"当前检索到的文档"、"已经调用过几次 Tool"、"用户是否已确认"） |
| 归属 | 是 State 的一个子集 | 是更大的容器，Memory 只是其中一个字段 |

Memory 是 State 的一部分，但 State 通常比 Memory 大得多——这也是为什么第 10 节要专门用一整节讲 LangGraph 的 `State`，而不是简单说"LangGraph 也有 Memory"。

### Key Takeaway

模型没有记忆，"记住"永远是调用方把历史重新传回去的结果。短期记忆＝对话消息列表；长期记忆通常需要额外的存储（比如把关键信息写进向量库或数据库，下次检索出来再塞回 Prompt）——这一点又与第 9 节的 RAG 检索机制相通。

---

<a id="9"></a>
## 9. RAG —— 本仓库的核心内容

这是整个仓库真正扎实覆盖的部分。先看完整流程图，再逐段对照代码。

```text
Documents（原始数据：文本文件/网页/目录/结构化数据/PDF）
  │  documents.py
  ▼
Split（切分成小块）
  │  text_splitter.py
  ▼
Embedding（把每个 chunk 转成向量）
  │  embeddings.py
  ▼
Vector Store（把 Document + 向量存起来）
  │  embeddings_chroma.py / embedding_chroma_persistence.py
  ▼
Retriever（包装成标准 Runnable，可以 .invoke(query)）
  │  embedding_chroma_retrieval.py
  ▼
Relevant Documents（检索到的相关片段）
  │
  ▼
（缺失环节）Prompt 把检索结果拼进去 → LLM 生成答案
  │  rag_pipeline.py（目前只有注释掉的 stub）
  ▼
Answer
```

### 9.1 Data Ingestion —— `documents.py` ✅

**Concept: `Document`**

`Document` 是 LangChain 里最基础的数据容器，只有两个字段：

```python
from langchain_core.documents import Document
doc = Document(page_content="正文文本", metadata={"source": "从哪来的"})
```

- `page_content`：纯文本，是后面 Embedding/切分要处理的对象。
- `metadata`：一个自由字典，描述这段文本"从哪来"（文件路径、URL、页码、作者……），检索之后可以用它做过滤、引用溯源，但**不会**被塞进 Embedding 计算（只有 `page_content` 参与向量化）。

**Concept: 为什么放弃 `langchain_community` 的 Loader（⚠️ Legacy vs ✅ Recommended）**

`langchain-community` 里原本提供了 `TextLoader`、`WebBaseLoader`、`DirectoryLoader`、`PyPDFLoader` 这些开箱即用的加载器，但该包正在被 sunset（仓库 `documents.py` 顶部的注释直接引用了官方 issue）。仓库的做法是绕开这些封装，**直接用底层库自己拼 `Document`**：

| 数据源 | ⚠️ Legacy（`langchain_community`） | ✅ 仓库实际写法（直接用底层库） |
|---|---|---|
| 文本文件 | `TextLoader` | `pathlib.Path.read_text()` |
| 网页 | `WebBaseLoader` | `requests` + `BeautifulSoup` |
| 目录 | `DirectoryLoader` | 手写生成器 `lazy_load_directory`（`yield`，惰性加载） |
| PDF | `PyPDFLoader` | `pypdf.PdfReader`，一页一个 `Document` |

以 `load_text_file()` 为例：

```python
text = temp_file_path.read_text()
doc = Document(page_content=text, metadata={"source": str(temp_file_path)})
```

这就是 `TextLoader` 内部本来在做的事——`langchain-community` 的加载器从来都不是"魔法"，只是对这几行代码的封装。理解了这一点，你会发现**自己实现一个 Loader 并不难**，只要最终产出 `Document` 对象即可，这也是仓库 `CLAUDE.md` 里明确要求"新增 loader 类样例时遵循同样模式"的原因。

`lazy_load_directory` 值得单独讲一下 **Lazy Loading** 这个概念：

```python
def lazy_load_directory(directory):
    for path in sorted(Path(directory).glob("*.txt")):
        yield Document(page_content=path.read_text(), metadata={"source": str(path)})
```

用 `yield` 而不是 `return [...]`，意味着文件是"调用方每要一个，才读一个"，而不是一次性把整个目录都读进内存——当目录里有几千个文件、或者文件很大时，这个区别直接决定程序会不会 OOM。这对应官方 Loader 接口里 `load()`（急切加载，返回 list）与 `lazy_load()`（惰性加载，返回 generator/iterator）的区别。

`load_pdf()` 展示了"一页一个 `Document`"的设计：

```python
for page_number, page in enumerate(reader.pages):
    doc = Document(
        page_content=page.extract_text(),
        metadata={"source": str(pdf_path), "page": page_number},
    )
```

为什么不是"整份 PDF 一个 Document"？因为 RAG 检索的最终目的是"精确定位到答案所在的那一小段文本"，如果一份 100 页的 PDF 只生成 1 个 Document，那么这个 Document 大概率会在切分阶段（下面 9.2 节）被机械地切成若干块，丢失"这段话来自第几页"这种有价值的 metadata；而一页一个 Document，配合切分时保留 metadata（`split_documents` 会做的事），最终每个 chunk 都能追溯到具体页码，方便引用溯源。

`load_structure()` 展示了 **Document 不一定来自"加载器"**：任何能产出文本 + 描述性字段的代码都可以手动构造 `Document`，比如从数据库查询结果、从 JSON API 响应直接构造，不需要绕道一个"Loader"类。

### 9.2 Indexing —— `text_splitter.py` ✅

**Concept: 为什么不能把整个 Document 直接塞进 Prompt**

两个原因：（1）模型的上下文窗口和 Embedding 模型的输入长度都有上限；（2）即使塞得下，太长的上下文会稀释真正相关的信息（"大海捞针"问题），检索质量反而下降。所以要先把长文档切成小块（chunk），再分别 Embedding、分别存储、分别检索。

**`CharacterTextSplitter` vs `RecursiveCharacterTextSplitter`**

仓库两个函数分别演示了这两种切分器，注释里已经把区别讲得很清楚：

```python
# CharacterTextSplitter：只认一个分隔符（这里是 "\n\n"）。
# 如果两个分隔符之间的内容本身还是超过 chunk_size，它不会继续往下切——
# 直接保留成一个超长 chunk。
splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)

# RecursiveCharacterTextSplitter：按一组分隔符的优先级依次尝试
# （默认 ["\n\n", "\n", " ", ""]，即先按段落、再按行、再按词、最后按字符），
# 直到每个 chunk 都不超过 chunk_size 为止 —— 这也是"recursive"这个名字的来源。
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
```

`chunk_overlap` 是另一个关键参数：让相邻 chunk 之间重叠一部分内容，避免"一句话刚好被切成两半，两个 chunk 单独看都丢了关键信息"。

**为什么 `RecursiveCharacterTextSplitter` 是通用推荐（✅ Recommended）**

`CharacterTextSplitter` 在 `split_by_character()` 的示例里就露出了它的局限：第二段文字本身超过 100 字符，又没有内部的 `\n\n`，于是被原样保留成一个超长 chunk（注释里明确写了"unlike CharacterTextSplitter's single long chunk 2"）。`RecursiveCharacterTextSplitter` 会在这种情况下继续往下退化到按行、按词切，尽量让每个 chunk 都符合 `chunk_size` 的约束——这就是为什么它是纯文本场景下的默认推荐选择。

**`split_documents` vs `split_text`**

```python
chunks = splitter.split_text(text)          # 输入/输出都是字符串，没有 metadata
chunks = splitter.split_documents([doc])    # 输入/输出都是 Document，metadata 会被复制到每个 chunk 上
```

结合 9.1 节 PDF 一页一个 Document 的设计，`split_documents` 是把"这段 chunk 来自第几页/哪个文件"这条信息一路带下去的关键一环。

### 9.3 Embedding —— `embeddings.py` ✅

**Concept: Embedding 是什么**

Embedding 是把一段文本转成一个固定长度的浮点数向量，语义相近的文本，向量在空间中的距离也相近——这是语义检索（而不是关键词匹配）的数学基础。

```python
from langchain_aws import BedrockEmbeddings
vector = embeddings.embed_query(text)          # 单条文本，用于查询
vectors = embeddings.embed_documents(texts)     # 批量文本，用于建库
```

- `embed_query` 和 `embed_documents` 是**两个不同方法**，即使很多模型内部实现完全一样——保留这个区分是因为有些 Embedding 模型对"查询文本"和"被检索文本"会做不同的预处理（例如加不同的前缀 instruction），LangChain 的接口设计提前把这个可能性纳入了考虑。

**本地缓存的工程实践**

```python
def _cache_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest()
```

`embed_with_local_cache` 用文本+model_id 的哈希作为 key，把向量缓存进本地 JSON 文件——这不是 LangChain 内置功能，是仓库自己加的一层。**为什么要缓存**：Embedding 调用是按量计费/有速率限制的外部 API 调用，同一段文本重复 Embedding 是纯粹的浪费；哈希里带上 `model_id` 是因为换一个 Embedding 模型，同样文本产生的向量维度/数值都会完全不同，缓存 key 不带模型信息会导致"张冠李戴"。🏭 这是一个值得搬进生产系统的模式，正式项目通常会换成 Redis/数据库而不是单个 JSON 文件（并发写入 JSON 文件本身不是线程安全的）。

### 9.4 Vector Store —— `embeddings_chroma.py` / `embedding_chroma_persistence.py` ✅

**Concept: Vector Store**

Vector Store 把"Document 的文本 + 它的 Embedding 向量 + metadata"三者存在一起，并提供"给一个查询向量，返回最相近的 N 个 Document"的检索能力。仓库用的是 **Chroma**，一个可以嵌入式运行（不需要单独部署服务）、也可以持久化到磁盘的开源向量数据库。

```python
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory,
)
```

`from_documents` 一步做了两件事：调用 `embeddings.embed_documents()` 把每个 `Document.page_content` 转成向量，然后把 `(向量, page_content, metadata)` 一起写进 `persist_directory` 下的 SQLite 文件（`chroma.sqlite3`）。

**`similarity_search` vs `similarity_search_with_score`**

```python
results = vector_store.similarity_search(query, k=2)                 # 只要 Document
results = vector_store.similarity_search_with_score(query, k=2)      # 额外带距离分数
```

`with_score` 版本返回 `(Document, score)` 元组——score 是距离度量（Chroma 默认距离越小越相似），用来在业务代码里做"低于某个相似度阈值就不采纳"的过滤，避免检索到几乎不相关的内容也硬塞进 Prompt。

**Concept: 持久化（`embedding_chroma_persistence.py`）**

`embeddings_chroma.py` 用的是临时目录（`tempfile.mkdtemp()`），每次运行都从零 Embedding、跑完就删——**只证明了 Chroma 在单次进程内能用，没有证明"持久化"**。`embedding_chroma_persistence.py` 专门用一个稳定路径（`.chroma_persistence_demo`）来补上这一课：

```python
PERSIST_DIRECTORY = Path(__file__).parent / ".chroma_persistence_demo"

def has_persisted_data(persist_directory: Path) -> bool:
    return (persist_directory / "chroma.sqlite3").exists()
```

- 第一次运行：`chroma.sqlite3` 不存在 → 调用 `build_vector_store`（即 `Chroma.from_documents`）→ 对每个 Document 发起一次真实的 Embedding 调用 → 写入磁盘。
- 第二次运行：`chroma.sqlite3` 已存在 → 走 `load_existing_vector_store`：

```python
return Chroma(
    embedding_function=embeddings,
    persist_directory=str(persist_directory),
)
```

注意这里**没有传 `documents`/`texts` 参数**——这是与 `from_documents` 的关键区别：直接用构造函数 + `persist_directory` 打开一个已存在的集合，不会触发任何新的 Embedding 调用，只是把磁盘上已经算好的向量加载回来。之后调用 `similarity_search` 时，只有**查询本身**会被 Embedding，被检索的历史 Document 向量是直接从磁盘读的。

这里还有一个容易被忽略但很重要的细节：两条路径都没有显式传 `collection_name`，因此都落在默认值 `"langchain"` 上——这保证了"建库"和"读库"两次调用能对上同一个集合，如果哪天要在同一个 `persist_directory` 里存多个不相关的集合，就必须显式指定不同的 `collection_name`，否则会互相覆盖/混淆。

**Windows 上的一个工程细节**（`utils/chroma_temp_dir.py`）：

```python
path = tempfile.mkdtemp()
try:
    yield path
finally:
    shutil.rmtree(path, ignore_errors=True)
```

没有用 `tempfile.TemporaryDirectory()` 自带的上下文管理器，是因为 Chroma 在进程存活期间会一直持有 `chroma.sqlite3` 的文件句柄，Windows 不允许删除一个被占用的文件，`TemporaryDirectory` 退出时自动清理会因此抛 `PermissionError`。手动 `shutil.rmtree(..., ignore_errors=True)` 是绕开这个问题的折中方案（忽略清理失败，而不是让整个脚本崩溃）。**这是一条非常真实的"生产踩坑"记录**，值得记住：任何在 Windows 上用临时目录 + 长期持有文件句柄的库，都要留意这个坑。

### 9.5 Retrieval —— `embedding_chroma_retrieval.py` ✅

**Concept: `as_retriever()`**

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(query)
```

`as_retriever()` 把一个 `VectorStore` 包装成 `VectorStoreRetriever`——一个标准 `Runnable`。注意仓库注释里强调的一点：`as_retriever()` 本身**不执行任何检索**，它只是返回一个"关联了 `vector_store` 和这些 `search_kwargs` 的对象"，真正的 embed-query + 搜索发生在之后调用 `.invoke(query)` 的那一刻（延迟执行）。

**为什么要多包一层 Retriever，而不是直接调用 `vector_store.similarity_search()`？** 因为 `Retriever` 是标准 `Runnable`，可以直接用 `|` 拼进 LCEL 链条（`retriever | format_docs | prompt | model`），而 `vector_store.similarity_search()` 是 `VectorStore` 这个类特有的方法，不满足 Runnable 接口，没法直接参与链式组合。这正是第 4 节讲的"统一接口带来可组合性"的一个具体例子。

**Concept: Similarity Search vs MMR**

```python
# 纯相似度：只看"离 query 有多近"
return vector_store.as_retriever(search_kwargs={"k": 2})

# MMR（Maximal Marginal Relevance）：
# 先按相似度捞 fetch_k 个候选（比最终要的 k 个多），
# 再贪心地选 k 个——每一步都在"离 query 近"和"和已选结果不重复"之间做权衡
return vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 4},
)
```

MMR 解决的问题是：纯相似度检索的 top-k 结果经常互相高度重复（想象一个知识库里有 10 篇几乎一样的文章，相似度检索可能把这 10 篇全部推到前几名），MMR 用"边际相关性"惩罚与已选结果过于相似的候选，让最终结果集覆盖更多不同的角度。仓库的注释也诚实地指出：因为示例只有 4 篇主题完全不同的 Document，冗余本来就很少，所以这个例子里 MMR 和纯相似度结果恰好一样——**这是一个很好的提醒：功能上的差异不一定在每个数据集上都能观察到，理解原理比死记"用了就一定不同"更重要**。

### 9.6 Generation —— `rag_pipeline.py` ⚠️ 未实现

这是 RAG 流程里唯一缺失的一环。目前整个文件都是注释掉的 stub：

```python
# def create_kb():
#     """Create a vector store from knowledge base"""
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     ...
# def demo_basic_rag():
#     pass
# def format_docs(docs):
#     pass
```

补完它需要把本节前面的所有组件（Document → Splitter → Embeddings → Chroma → Retriever）与第 3、4 节的 Prompt Template、LCEL 组合起来，再加一步"把检索到的多个 Document 拼成一段字符串塞进 Prompt"。一个完整、但仍然简单的实现（第 14 节 Example 4 会给出完整可跑的版本）：

```python
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only using the provided context. If the context "
               "doesn't contain the answer, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

answer = rag_chain.invoke("What LLM providers does Bedrock support?")
```

这一行 `{"context": retriever | format_docs, "question": RunnablePassthrough()}` 其实就是 `RunnableParallel` 的字典简写形式：`context` 这一路先检索、再拼成字符串；`question` 这一路原样透传；两路的结果汇总成一个 dict 送进 `prompt`。这也正是"RAG 生成半段"和第 4 节 LCEL 概念的交汇点。

---

<a id="10"></a>
## 10. LangGraph 是什么

### 📘 本仓库完全未覆盖 —— 全部为补充知识

### 10.1 为什么已经有 LangChain / Agent，还需要 LangGraph

对比三种编排方式：

```text
Simple Chain（第 4 节 LCEL）
A → B → C                         固定顺序，没有分支和循环

Agent（第 7 节）
LLM → Tool → LLM → Tool → LLM     由模型自己决定路径，但整个循环是一个黑盒，
                                   你很难在中间插入"必须先做 X 再做 Y"这种硬约束

LangGraph
       ┌───────┐
       ▼       │
START → A → B ─┘
       ▼
       C
       ▼
      END
```

LangGraph 把"下一步走哪"从"模型脑子里的隐式决策"变成"图上一条显式的边（可以是固定边，也可以是条件边）"。它解决的是这些 Agent 循环天然不擅长的问题：

- **Stateful workflows**：整个流程共享一份显式的 `State`，每一步读/写这份状态，而不是隐式塞在一堆消息里。
- **Cycles（循环）**：图允许一个节点的出边指回自己或指回更早的节点——这正是"Agent Loop"背后真正的实现机制。
- **Branching / Conditional execution**：根据当前状态动态决定走哪条边。
- **Human-in-the-loop**：可以在图执行到某个节点后暂停，等人工审批/输入后再继续。
- **Checkpointing / Persistence**：每一步状态变化都可以落盘，进程重启、几小时后回来，都能从中断点继续。
- **Multi-agent workflow**：多个"节点"各自是一个独立的（子）Agent，通过共享/传递 State 协作。

### 10.2 与仓库现状的连接点

仓库现在连"检索之后用 Prompt+LLM 生成答案"（LCEL 级别的编排）都还没做，直接跳到 LangGraph 有点"没学走就想跑"。但理解 LangGraph 的价值在于：一旦你把 `rag_pipeline.py` 补完，很自然会遇到下一个问题——"如果检索到的内容不够回答问题，要不要换个关键词重新检索一次？最多重试几次？"——这种"根据结果决定是否要绕回去重做一步"的需求，正是 LangGraph 的用武之地，而不是继续在线性 LCEL 链条里硬塞 if/else。

---

<a id="11"></a>
## 11. LangGraph 核心概念：State / Node / Edge

### 11.1 State 🔑

```python
from typing import TypedDict

class State(TypedDict):
    messages: list
    retry_count: int
```

`State` 是整张图在执行过程中传递的**共享数据结构**（通常是一个 `TypedDict` 或 Pydantic Model）。每个节点函数接收当前 `State`，返回一个"要合并进 State 的增量更新"（不是必须返回完整 State——具体字段是"覆盖"还是"累加"取决于该字段是否配置了 `reducer`，例如 `messages` 字段常用 `add_messages` reducer 做"追加"而不是"覆盖"）。为什么需要它：图上每个节点都是独立、可能异步执行的单元，State 是它们之间唯一的通信媒介，没有 State，节点之间就没有办法传递信息。

### 11.2 Node

```python
def chatbot(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```

Node 是一个普通函数（或 Runnable），输入是当前 `State`（或它的一部分），输出是"要更新到 State 里的字段"。这里 `chatbot` 节点读取 `state["messages"]`，调用模型，把新的 `AIMessage` 追加回去。

### 11.3 Edge

```python
graph.add_edge("A", "B")   # 固定边：A 执行完之后一定去 B
```

```python
graph.add_conditional_edges(
    "agent",
    should_continue,        # 一个函数：接收 State，返回下一个节点的名字
    {"continue": "tools", "end": END},
)
```

固定边（Edge）表示"确定性流转"；条件边（Conditional Edge）由一个路由函数在运行时根据当前 `State` 决定走哪条分支——这是 LangGraph 把"分支逻辑"从模型的隐式决策，变成代码里显式可读、可测试的路由函数的核心机制。

### 11.4 START / END

```text
START
  │
  ▼
Node
  │
  ▼
Node
  │
  ▼
END
```

`START` 和 `END` 是 LangGraph 内置的两个特殊标记节点，分别表示"图的入口"和"图执行结束"。每张图必须能从 `START` 到达至少一条通往 `END` 的路径，否则图会一直循环下去。

### 11.5 一个完整的条件边示例

```text
User
 ▼
LLM
 ▼
Need Tool?
 ├── Yes → Tool → LLM（回到 LLM，可能再次判断是否还需要 Tool）
 └── No  → END
```

对应代码骨架（示范，非仓库代码）：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END

builder = StateGraph(State)
builder.add_node("agent", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")   # 执行完 Tool 之后回到 agent 节点——这就是"循环"

graph = builder.compile()
```

`add_edge("tools", "agent")` 这一行就是"环"（cycle）在代码里的样子——`agent → tools → agent → tools → ...` 直到 `should_continue` 返回 `END`。

---

<a id="12"></a>
## 12. LangGraph Agent 完整示例

```text
START
  │
  ▼
Agent Node（调用 LLM，可能产生 tool_calls）
  │
  ▼
Should call tool?
 ├── No ──────────────→ END
 │
 Yes
  │
  ▼
Tool Node（实际执行 Tool，把结果包装成 ToolMessage）
  │
  ▼
Agent Node（带着 Tool 结果，再次调用 LLM）
  │
  ▼
  ...（回到"Should call tool?"，循环）
```

完整可读的代码（⭐ Advanced，示范用途，非仓库代码）：

```python
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    # add_messages 是一个 reducer：节点返回 {"messages": [new_msg]} 时，
    # 会把 new_msg "追加"到已有列表末尾，而不是整体覆盖。
    messages: Annotated[list, add_messages]


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city} is sunny, 25 degrees."


tools = [get_weather]
model_with_tools = model.bind_tools(tools)   # 让模型知道有哪些 Tool 可用


def agent_node(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # AIMessage.tool_calls 非空，说明模型这一步决定要调用 Tool
    return "tools" if getattr(last_message, "tool_calls", None) else "end"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "What's the weather in Sydney?"}]})
print(result["messages"][-1].content)
```

逐段说明：

- **Agent Node**：把当前所有消息喂给"绑定了 Tool 的模型"（`model.bind_tools(tools)`），模型要么直接回答，要么在 `AIMessage.tool_calls` 里写下"我要调用 `get_weather(city="Sydney")`"。
- **Tool Node**：LangGraph 内置的 `ToolNode` 会自动读取上一条 `AIMessage` 里的 `tool_calls`，执行对应的 Python 函数，把结果包装成 `ToolMessage` 加回 `messages`——省去了你手写"解析 tool_calls → 执行 → 包装成 ToolMessage"这段样板代码。
- **State**：`messages` 是唯一贯穿全图的字段，`add_messages` reducer 保证每个节点返回的新消息是"追加"而不是"覆盖掉之前的对话历史"。
- **Conditional Edge**：`should_continue` 检查最新一条消息有没有 `tool_calls`，决定是回到工具执行分支还是结束。
- **Graph execution**：`graph.invoke(...)` 从 `START` 开始，沿着 `agent ⇄ tools` 反复流转，直到 `should_continue` 返回 `"end"`。

这段代码本质上就是 `langchain.agents.create_agent`（第 7.3 节）在底层帮你搭好的东西——理解了上面这段手写版本，再回头看 `create_agent`，就能明白它省掉的正是这整套 `StateGraph` 搭建过程。

---

<a id="13"></a>
## 13. LangChain vs LangGraph

| | LangChain | LangGraph |
|---|---|---|
| 主要用途 | 提供标准化组件（Model/Prompt/Tool/Retriever）与简单编排（LCEL） | 编排复杂、有状态、可能带环的工作流 |
| Chain | ✅ 核心能力（`\|` / LCEL） | 可以在节点内部使用 LCEL Chain |
| Agent | ✅ `create_agent`（1.0 起，内部基于 LangGraph） | ✅ 更底层、更可控的图形式 Agent |
| State | 隐式（消息历史） | ✅ 显式 `TypedDict`/Pydantic State，任意字段 |
| Workflow | 线性为主 | ✅ 任意有向图，含分支/环 |
| Loop | Agent 内部隐式循环，不易插入自定义控制 | ✅ 显式的边，循环一目了然、可插入自定义条件 |
| Conditional logic | 需要手写 `RunnableBranch` 等变通方案 | ✅ 原生 `add_conditional_edges` |
| Human approval | 无原生支持 | ✅ `interrupt` / checkpointer 原生支持 |
| Persistence | 无原生支持 | ✅ `checkpointer`（可存内存/SQLite/Postgres 等） |
| Multi-agent | 需要手工拼装 | ✅ 天然适合（每个子图/节点是一个 Agent） |
| 学习难度 | 较低，接口统一、上手快 | 较高，需要理解图、状态、reducer 等概念 |

### 什么时候用 LangChain，什么时候用 LangGraph？

- 任务是"固定几步走完"（比如仓库里缺的那一段：检索 → 拼 Prompt → 调模型），用纯 LCEL 就够，不需要引入图的复杂度。
- 任务需要模型自主决定调用哪些 Tool、调用几次，但不需要人工审批、不需要复杂分支，`create_agent` 这一层高层封装通常已经够用。
- 任务需要**显式控制流程**（比如"检索结果不够好就必须重新检索，最多重试 3 次"）、需要**长时间运行并可中断续跑**、需要**人工审批关键步骤**、或者是**多个 Agent 协作**——这些都是 LangGraph 的用武之地。

### 两者是不是竞争关系？

不是。**LangGraph 建立在 LangChain 的 Model / Tool / Message 等组件之上**，是同一个生态里"更高阶的编排层"，而不是另起炉灶的竞品。事实上从 LangChain 1.0 开始，`create_agent` 本身就是"预先用 LangGraph 搭好的一张图"——这意味着你写的 LangChain 代码，运行时环境里几乎一定已经跑在 LangGraph 之上，只是被高层 API 隐藏掉了细节。一个常见、也是官方推荐的成长路径是：先用 LangChain 的高层抽象（Chain / `create_agent`）快速搭出可用的东西，当遇到高层抽象表达不了的控制需求时，再"下沉"到 LangGraph 的 `StateGraph` 拿到完整控制权。

---

<a id="14"></a>
## 14. 从简单到复杂的 5 个项目

以下 5 个例子按复杂度递增排列，Example 1–3 是通用官方概念的最小实现，Example 4–5 **直接建立在你仓库现有代码之上**，是把 `rag_pipeline.py` 补完、再进化成 Agent 的实操路径。

### Example 1 —— Simple LLM

```text
User → Prompt → LLM → Answer
```

- **使用场景**：单轮问答，没有变量、没有工具、没有记忆。
- **架构**：`ChatModel.invoke()` 直接调用。
- **完整代码**（等价于仓库 `check_connection.py` 的核心逻辑）：

```python
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

llm = ChatBedrockConverse(model="amazon.nova-micro-v1:0", region_name="ap-southeast-2")
response = llm.invoke([HumanMessage(content="What is LangChain?")])
print(response.content)
```

- **执行流程**：构造消息 → `.invoke()` 发起一次网络请求 → 返回 `AIMessage`。
- **关键知识点**：Chat Model 是 Runnable；输入是消息列表不是裸字符串。
- **常见错误**：把 `response` 当字符串直接拼接（要用 `.content`）；忘记处理 `ClientError`（仓库 `check_connection.py` 已经正确处理了这一点）。
- **扩展方向**：加 `SystemMessage`；加 `temperature`/`max_tokens` 配置。

### Example 2 —— Chain

```text
Input → Prompt → LLM → Parser
```

- **使用场景**：需要变量化 Prompt，并且要拿到干净的字符串而不是 `AIMessage` 对象。
- **架构**：LCEL `|` 三段式。
- **完整代码**：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "Explain {topic} in one sentence."),
])
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "LangChain"})
print(result)   # 直接是字符串，不用再取 .content
```

- **执行流程**：`{"topic": "LangChain"}` → Prompt 填充变量 → 生成消息列表 → 模型生成 `AIMessage` → `StrOutputParser` 提取 `.content`。
- **关键知识点**：`|` 把三个 Runnable 串成一个新 Runnable；`StrOutputParser` 是最简单的 Output Parser。
- **常见错误**：Prompt 里变量名和 `.invoke()` 传的 key 对不上，报 `KeyError`。
- **扩展方向**：换成 `with_structured_output` 拿到 Pydantic 对象而不是字符串。

### Example 3 —— Tool Calling

```text
User → LLM → Tool → LLM → Answer
```

- **使用场景**：需要模型"做事"而不只是"说话"（查天气、算数、查库存）。
- **架构**：`bind_tools` + 手动执行循环（比 Example 5 的 LangGraph 版本更"手工"，用来理解底层机制）。
- **完整代码**：

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

llm_with_tools = llm.bind_tools([add])
messages = [HumanMessage(content="What is 12 plus 30?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    result = add.invoke(tool_call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

final = llm_with_tools.invoke(messages)
print(final.content)
```

- **执行流程**：模型第一次调用产生 `tool_calls` → 代码手动执行 `add` → 包装成 `ToolMessage` 追加进历史 → 第二次调用模型，这次它能看到工具结果，给出最终答案。
- **关键知识点**：`tool_call_id` 必须和请求对上；一次模型响应里可能有多个 `tool_calls`（示例里用 `for` 循环处理，即使这里只有一个）。
- **常见错误**：只调用一次模型就结束，没有把 Tool 结果喂回去做第二次调用。
- **扩展方向**：把这段手写循环换成第 12 节的 `StateGraph`（LangGraph 会自动处理这个循环）。

### Example 4 —— RAG（补完仓库的 `rag_pipeline.py`）

```text
User
 ▼
Retriever（复用 embedding_chroma_retrieval.py 的检索逻辑）
 ▼
Prompt（拼接检索结果 + 用户问题）
 ▼
LLM
 ▼
Answer
```

- **使用场景**：回答仓库知识库范围内的问题，正是你仓库缺的最后一环。
- **架构**：`RunnableParallel` + LCEL。
- **完整代码**（直接复用仓库已有的 `build_sample_documents` / `build_vector_store` / `temp_persist_directory` / `build_embeddings_client`）：

```python
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_aws import ChatBedrockConverse

from config import load_config
from embeddings_chroma import build_sample_documents, build_vector_store
from utils.chroma_temp_dir import temp_persist_directory
from utils.embeddings_client import build_embeddings_client


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    config = load_config()
    embeddings = build_embeddings_client(config)
    llm = ChatBedrockConverse(
        model=config.model.bedrock_model_id,
        region_name=config.aws.region,
        aws_access_key_id=config.aws.access_key_id,
        aws_secret_access_key=config.aws.secret_access_key,
        aws_session_token=config.aws.session_token,
    )

    with temp_persist_directory() as persist_directory:
        vector_store = build_vector_store(embeddings, build_sample_documents(), persist_directory)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only using the provided context. "
                       "If it doesn't contain the answer, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(rag_chain.invoke("What LLM providers does Bedrock support?"))


if __name__ == "__main__":
    main()
```

- **执行流程**：问题字符串同时进入两条路径——一条经过 `retriever` 检索、`format_docs` 拼接成一段文本；另一条被 `RunnablePassthrough()` 原样保留——两路结果汇总成 `{"context": ..., "question": ...}`，喂给 `prompt`，再到 `llm`，最后 `StrOutputParser` 拿到纯文本答案。
- **关键知识点**：这是本总结第 4、9 节内容的直接汇合点；`retriever | format_docs` 说明 Retriever（Runnable）可以和普通函数（包装成 `RunnableLambda`，`format_docs` 会被 `|` 自动包装）继续用 `|` 连接。
- **常见错误**：忘记限定 Prompt "只用提供的上下文回答"，模型会用自己的训练知识回答，而不是真正基于检索结果——这是 RAG 项目最容易出现、也最难被发现的错误（表面上答案对了，但其实没有真正"基于你的数据"）。
- **扩展方向**：加上 `similarity_search_with_score` 的分数过滤，检索分数太差时直接告诉用户"知识库里没有相关内容"而不是让模型硬答。

### Example 5 —— LangGraph Agent（RAG + 自主重试）

```text
START
 ▼
Retrieve（检索）
 ▼
Grade（判断检索结果是否足够回答问题）
 ├── 足够 → Generate → END
 └── 不够 → Rewrite Query → 回到 Retrieve（循环，最多重试 N 次）
```

- **使用场景**：Example 4 的静态 RAG 链条有一个明显缺陷——不管检索到的内容好不好，都会硬生生地把它喂给模型。这个例子加入"自我评估 + 重试"，这正是第 10 节说的"LangGraph 能表达 LCEL 表达不了的控制流"的真实案例。
- **架构**：`StateGraph`，带一条回边（循环）。
- **完整代码**（⭐ Advanced，建立在 Example 4 的组件之上）：

```python
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    question: str
    documents: list
    answer: str
    retry_count: int


MAX_RETRIES = 2


def retrieve_node(state: RAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def grade_node(state: RAGState) -> str:
    # 简化版评分：真实项目通常会再调用一次 LLM 做相关性判断
    if state["documents"] and state["retry_count"] < MAX_RETRIES:
        return "generate"
    if state["retry_count"] >= MAX_RETRIES:
        return "generate"  # 重试次数用完，用现有结果硬生成，避免死循环
    return "rewrite"


def rewrite_node(state: RAGState) -> dict:
    new_question = llm.invoke(
        f"Rewrite this question to be more specific for search: {state['question']}"
    ).content
    return {"question": new_question, "retry_count": state["retry_count"] + 1}


def generate_node(state: RAGState) -> dict:
    context = format_docs(state["documents"])
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]}
    )
    return {"answer": answer}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", grade_node, {"generate": "generate", "rewrite": "rewrite"})
builder.add_edge("rewrite", "retrieve")   # 循环回检索节点
builder.add_edge("generate", END)

graph = builder.compile()
result = graph.invoke({"question": "...", "documents": [], "answer": "", "retry_count": 0})
print(result["answer"])
```

- **执行流程**：`retrieve` 检索 → `grade_node`（条件边路由函数，不是普通节点，不写入 State）判断够不够好 → 不够就 `rewrite`（重写问题）再回到 `retrieve`，`retry_count` 每次 +1，防止死循环 → 够了（或重试次数耗尽）就 `generate` 生成答案 → `END`。
- **关键知识点**：`retry_count` 这个"控制循环终止条件"的字段，正是第 11 节强调的"State 不只是消息历史，还包含业务变量"的具体体现；`grade_node` 作为条件边的路由函数，只读 State、不写 State（写 State 是普通 Node 的职责）。
- **常见错误**：忘记设置重试上限，条件边永远走 `"rewrite"` 分支，图无限循环下去（这是 LangGraph 新手最常踩的坑，第 18 节会专门讲怎么发现它）。
- **扩展方向**：加 `checkpointer` 做持久化，让这个图可以在长时间运行的服务里跨请求保留状态；把 `grade_node` 换成真正调用 LLM 做相关性判断（`with_structured_output` 返回一个 `{"is_relevant": bool}`）。

---

<a id="15"></a>
## 15. Sample Code 逐个分析

### 15.1 文件 × 概念映射表

| File | 关键 Class / Function | LangChain 概念 | LangGraph 概念 | Purpose |
|---|---|---|---|---|
| `config.py` | `Config`, `load_config()` | （无，纯配置管理） | — | 从环境变量加载 AWS 凭证与模型 ID，供其余脚本统一使用 |
| `documents.py` | `Document`, `load_text_file/web_content/lazy_loader/load_structure/load_pdf` | `Document`、Lazy Loading | — | 从文本/网页/目录/结构化数据/PDF 构建 `Document`，替代 `langchain_community` Loader |
| `text_splitter.py` | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter` | Text Splitter | — | 把长文本/`Document` 切成小 chunk |
| `embeddings.py` | `BedrockEmbeddings`, `embed_query/embed_documents` | Embeddings | — | 生成文本向量，附带本地 JSON 缓存 |
| `utils/embeddings_client.py` | `build_embeddings_client()` | Embeddings（工厂函数） | — | 统一构造 `BedrockEmbeddings` 客户端，供其它样例复用 |
| `embeddings_chroma.py` | `Chroma.from_documents`, `similarity_search[_with_score]` | Vector Store | — | 把 `Document` + 向量存入 Chroma，做相似度检索 |
| `utils/chroma_temp_dir.py` | `temp_persist_directory()` | （Vector Store 的辅助工具） | — | 提供一个 Windows 安全的临时目录上下文管理器 |
| `embedding_chroma_retrieval.py` | `as_retriever()`, `VectorStoreRetriever.invoke()` | Retriever、Runnable | — | 把 Vector Store 包装成标准 Runnable，对比相似度检索与 MMR |
| `embedding_chroma_persistence.py` | `Chroma(persist_directory=...)`（无 documents 参数） | Vector Store 持久化 | — | 演示 Chroma 集合跨进程运行持久化到磁盘 |
| `check_connection.py` | `ChatBedrockConverse`, `HumanMessage` | Chat Model | — | 验证 Bedrock 凭证/模型访问是否正常 |
| `rag_pipeline.py` | （全部注释掉） | RAG 生成端（尚未实现） | — | 预留的 RAG 完整流程入口，目前是 stub |

（"LangGraph 概念"一列全部为空——如前所述，仓库没有任何 LangGraph 代码，这份表格如实反映这一点，而不是强行凑内容。）

### 15.2 横向关系分析

- `config.py` 被除 `documents.py`、`text_splitter.py` 外的所有脚本导入——这两个文件之所以不需要配置，是因为它们完全不涉及网络调用（纯本地文本处理），这也印证了 README 里说的"这两个文件不需要任何配置即可运行"。
- `embeddings_chroma.py` 的 `build_sample_documents()` / `build_vector_store()` 被 `embedding_chroma_retrieval.py` 和 `embedding_chroma_persistence.py` 直接 `import` 复用，而不是各自重新定义——三个文件共享同一份示例数据，检索结果才具有可比性（这一点在源码注释里被反复强调）。
- `utils/embeddings_client.py` 和 `utils/chroma_temp_dir.py` 是两个从多个样例文件中提炼出来的公共工具，体现了"发现重复代码就提取公共函数"的基本工程原则。
- `rag_pipeline.py` 是唯一"断链"的文件——它在概念上应该消费 `text_splitter.py` + `embeddings_chroma.py` 的产出，但目前完全没有实现，是本总结第 14 节 Example 4 重点要补的部分。

### What / Why / How / Relationship / Alternative / Modern API（挑几个最重要的模块展开）

**`Document`（`langchain_core.documents.Document`）**
- What：包含 `page_content` + `metadata` 的最小文本容器。
- Why：给"任意来源的文本"提供一个统一的下游接口，Text Splitter / Embeddings / Vector Store 都只认这一种类型。
- How：任何代码只要产出这两个字段就能构造它，不依赖特定 Loader。
- Relationship：是 Text Splitter 的输入/输出类型、是 Embeddings 处理的对象、是 Vector Store 存储的单元。
- Alternative：直接用裸字符串/字典，但会失去 metadata 溯源能力和与其它组件的互操作性。
- Modern API：✅ 本身就是当前推荐写法。

**`RecursiveCharacterTextSplitter`**
- What：按分隔符优先级递归切分文本的切分器。
- Why：`CharacterTextSplitter` 遇到单一分隔符切不动的长文本会失效，`RecursiveCharacterTextSplitter` 兜底解决这个问题。
- How：依次尝试 `["\n\n", "\n", " ", ""]`，直到每块都不超过 `chunk_size`。
- Relationship：上游接 `Document`/字符串，下游接 Embeddings。
- Alternative：`TokenTextSplitter`（按 token 数而不是字符数切，和模型上下文窗口对齐更精确）、语义切分（按 Embedding 相似度找自然断点，成本更高但边界质量更好）。
- Modern API：✅ 当前推荐，仓库用法正确。

**`Chroma`（`langchain_chroma.Chroma`）**
- What：本地/嵌入式运行的开源向量数据库封装。
- Why：需要"存向量 + 按相似度查"的能力，又不想额外运维一个独立服务。
- How：`from_documents` 建库并写盘；直接构造 `Chroma(persist_directory=...)` 打开已有库。
- Relationship：接收 `Embeddings` 客户端做向量化，产出可以 `as_retriever()` 的对象。
- Alternative：生产环境常见的托管方案有 Pinecone、Weaviate、pgvector、OpenSearch（当数据量大、需要多实例共享访问时，本地文件型的 Chroma 不是最佳选择）。
- Modern API：✅ `langchain_chroma` 是当前推荐包（⚠️ 注意不要从已废弃路径 `langchain.vectorstores.Chroma` 导入）。

---

<a id="16"></a>
## 16. 知识体系重组

不按文件顺序，而是按概念层级重新组织：

```text
LangChain
│
├── Models
│   └── ChatBedrockConverse ✅
│
├── Messages
│   └── HumanMessage ✅ / SystemMessage 📘 / AIMessage ✅(返回值) / ToolMessage 📘
│
├── Prompts 📘
│   └── ChatPromptTemplate
│
├── Runnables 📘（概念已隐含在 ✅ retriever.invoke() 里）
│   ├── Sequence（`|`)
│   ├── Parallel
│   └── Lambda
│
├── Output Parsers 📘
│   └── StrOutputParser / with_structured_output
│
├── Tools 📘
│   └── @tool / bind_tools
│
├── Agents 📘
│   └── create_agent（内部 = LangGraph 图）
│
├── Memory / State 📘
│   └── 消息历史 = messages 字段
│
└── Retrieval ✅（仓库核心）
    ├── Documents ✅        documents.py
    ├── Text Splitters ✅   text_splitter.py
    ├── Embeddings ✅       embeddings.py
    ├── Vector Stores ✅    embeddings_chroma.py / embedding_chroma_persistence.py
    └── Retrievers ✅       embedding_chroma_retrieval.py

RAG（组合以上组件） —— 检索半段 ✅ 完整，生成半段 ⚠️ 未实现（rag_pipeline.py）

LangGraph 📘（仓库完全未覆盖）
│
├── State
├── Nodes
├── Edges / Conditional Edges
├── Loops（循环边）
├── Checkpoints / Persistence
├── Human-in-the-loop
└── Multi-Agent
```

---

<a id="17"></a>
## 17. 常见误区（15 条）

1. ❌ **LangChain 就是一个 LLM**
   ✅ LangChain 是围绕 LLM 的一套组件/接口标准库，它本身不训练、不托管模型，只是统一了调用各家模型的方式。
   💡 因为初学者接触的第一个对象就是 `ChatModel`，容易把"框架"和"模型"混为一谈。

2. ❌ **Agent 和 Chain 是一样的东西**
   ✅ Chain 是写代码时就固定好的步骤序列；Agent 是运行时由模型自主决定步骤的循环。
   💡 两者最终都产出"输入 → 若干步骤 → 输出"，表面形态相似，但决策权在谁手上完全不同。

3. ❌ **LangGraph 可以完全替代 LangChain**
   ✅ LangGraph 建立在 LangChain 的 Model/Tool/Message 组件之上，是编排层而不是替代品；`create_agent` 本身内部就是用 LangGraph 搭的图。
   💡 两者经常被拿来对比，容易被理解成"二选一"的竞品关系。

4. ❌ **Tool 本身会进行 reasoning（推理判断）**
   ✅ Tool 只是一个被动执行的函数，"要不要调用、调用哪个"完全是 LLM 的决策，Tool 自己不会"思考"。
   💡 因为 Tool 常被叫做"Agent 的能力"，容易让人以为 Tool 也有智能。

5. ❌ **Memory 等于数据库**
   ✅ 最基础的 Memory 只是"每次请求把历史消息列表重新发一遍"，模型本身不持久化任何东西；持久化是调用方自己实现（或用 LangGraph 的 checkpointer）的。
   💡 "记忆"这个词天然让人联想到存储系统。

6. ❌ **RAG 等于 Vector Database**
   ✅ Vector Store 只是 RAG 检索半段里的一环，RAG 完整流程还包括 Document 加载、切分、Embedding、（更关键的）拼 Prompt 让 LLM 生成答案——仓库本身就是活生生的反例：有 Vector Store，但 RAG 生成半段还没实现，不能说这个仓库"已经有 RAG"。
   💡 很多教程把"向量库检索 Demo"直接叫作"RAG"，省略了生成这一步。

7. ❌ **Embedding 等于 LLM**
   ✅ Embedding 模型和 Chat/生成模型是两类不同的模型，前者输出向量、不生成文本；很多场景下二者甚至来自不同的模型家族（仓库用的 `amazon.titan-embed-text-v2:0` 和 `amazon.nova-micro-v1:0` 就是两个完全不同的模型）。
   💡 都叫"模型"、都通过同一个 Bedrock 客户端家族访问，容易被当成一回事。

8. ❌ **Agent 每一步都是确定的（deterministic）**
   ✅ 只要循环里有 LLM 参与决策，同样的输入在不同次运行可能走不同的路径（除非把 `temperature` 设为 0 且模型本身保证确定性，但多数供应商并不严格保证）。
   💡 写代码的人习惯了程序是确定性的，容易忽视"这一步是模型在做判断"这件事。

9. ❌ **Graph 就是普通 workflow（跟画个流程图一样）**
   ✅ LangGraph 的 Graph 有显式的 `State` 传递机制、reducer 合并规则、checkpoint 持久化能力，这些是普通"画个箭头图"的流程图工具不具备的。
   💡 图的可视化外观确实和流程图很像，容易忽视背后的状态管理机制。

10. ❌ **State 和 Memory 完全一样**
    ✅ Memory（对话历史）只是 State 里的一个字段，State 通常还包含业务变量（重试次数、当前阶段、待审批标记等），第 8.3 节有详细对比。
    💡 大多数入门 demo 的 State 里唯一有意义的字段就是 `messages`，让人误以为二者等价。

11. ❌ **Prompt Template 只是"字符串格式化"的花哨写法**
    ✅ 它额外提供了多角色消息组装、变量校验、与其它 Runnable 组合的能力，这些是纯 f-string 做不到或做起来很别扭的。
    💡 从最终效果（填充变量生成一段文本）上看确实很像 `.format()`。

12. ❌ **`langchain-community` 里的 Loader 是"官方推荐、永远能用"的标准做法**
    ✅ 该包正在被 sunset（仓库 `CLAUDE.md`/`documents.py` 明确提到），新代码应该直接用底层库构造 `Document`，或者使用各自维护的独立包（如 `langchain-chroma`、`langchain-aws`）里的组件。
    💡 大量教程和历史博客还在用 `from langchain.document_loaders import TextLoader` 这种旧写法。

13. ❌ **只要调用了 `.invoke()` 就说明这是"标准接口"，可以随便乱连**
    ✅ 两个 Runnable 能用 `\|` 连接，前提是前一个的**输出类型**要匹配后一个的**输入类型**（比如 `retriever` 输出 `list[Document]`，不能直接接一个期望字符串输入的 Runnable，中间要有 `format_docs` 这类转换）。
    💡 `.invoke()` 这个方法名在所有 Runnable 上都存在，容易让人误以为输入输出也都兼容。

14. ❌ **Similarity Search 的 Top-K 结果一定是"最相关"的**
    ✅ 它只是"向量距离最近"，向量距离和"人类认为的语义相关性"并不总是完全对齐，尤其当 chunk 切分不合理、或者查询表述与文档表述用词差异很大时。
    💡 "相似度"这个词听起来就等于"正确答案"，但它衡量的是向量空间距离，不是事实正确性。

15. ❌ **只要用了 LangGraph，就自动获得了持久化/断点续跑能力**
    ✅ 必须显式配置 `checkpointer`（比如 `InMemorySaver`、`SqliteSaver`、`PostgresSaver`）并在 `compile()` 时传入，`StateGraph` 默认不带持久化。
    💡 很多教程示例只展示了 `graph.invoke()` 一把梭，没有强调 checkpointer 是可选、需要显式开启的组件。

---

<a id="18"></a>
## 18. Debugging / Troubleshooting

- **查看实际发给模型的消息**：在调用 `.invoke()` 前 `print(messages)` 或者 `print(prompt.invoke(vars).to_messages())`——很多"模型答非所问"的问题，根源是 Prompt Template 填充出来的内容和你以为的不一样。
- **查看 Tool Call**：`ai_msg.tool_calls` 直接打印出来看模型到底想调用哪个函数、传了什么参数——参数类型不对、函数名拼写错都会在这里第一时间暴露。
- **查看 Graph State**：`graph.invoke(...)` 的返回值就是最终 `State`；调试中间过程可以用 `graph.stream(...)`，它会在每个节点执行完后 yield 当前状态快照，比只看最终结果更容易定位是哪一步出的问题。
- **Debug Agent Loop**：给 Agent Loop 加一个硬性最大迭代次数（LangGraph 的 `recursion_limit` 参数，或者第 14 节 Example 5 里手写的 `retry_count`），出问题时优先怀疑"死循环"而不是"逻辑错误"。
- **Debug RAG 检索质量**：用 `similarity_search_with_score` 而不是 `similarity_search`，把分数打印出来——如果 top 结果分数都很差（距离很大/相似度很低），问题往往出在 chunk 切分策略或者 Embedding 模型选择上，而不是 Prompt 写得不好。
- **发现 Prompt 问题**：把同一个 Prompt Template 拿到模型 Playground（比如 Bedrock 控制台）里手动跑一遍，排除是"这句话本身让模型困惑"还是"代码传参有 bug"。
- **发现 Tool Schema 问题**：检查 `tool.args_schema`（`@tool` 装饰后自动生成）是否符合预期——docstring 缺失或类型标注缺失，是模型"不知道怎么调用"的最常见原因。
- **发现 State 更新问题**：确认字段有没有配置正确的 reducer——普通字段默认是"整体覆盖"，只有配置了 `Annotated[..., add_messages]` 之类的字段才是"追加"，混淆这两种语义是新手最常见的 State 相关 bug。
- **避免无限循环**：任何带条件边循环回自身/更早节点的图，务必设一个显式的终止条件（重试计数、超时、或 `recursion_limit`），第 14 节 Example 5 的 `retry_count` 就是这个模式的最小实现。
- **仓库特有的调试点**：`embedding_chroma_persistence.py` 判断"是否已有持久化数据"是靠检查 `chroma.sqlite3` 文件是否存在——如果你改了 `PERSIST_DIRECTORY` 或者手动删了文件却发现程序行为没变，先确认是不是路径没对上，而不是代码逻辑错了。

---

<a id="19"></a>
## 19. Production Considerations

### Reliability
- 🏭 Retry / Timeout：外部 API 调用（Bedrock、向量库网络请求）都应该配重试和超时，仓库目前 `check_connection.py` 只捕获了 `ClientError` 并打印，没有重试逻辑——生产环境通常需要指数退避重试。
- 🏭 Fallback：主模型不可用时切换到备用模型/备用区域，`ChatBedrockConverse` 的统一接口让这种切换代码量很小。

### Cost
- 🏭 Token 用量：`AIMessage.usage_metadata` 里带 token 消耗，值得在生产系统里做日志/监控。
- 🏭 模型选择：`config.py` 里 `amazon.nova-micro-v1:0` 是一个轻量级模型的默认选择，体现了"默认给便宜的模型，需要更强能力时再显式配置"的成本意识。
- 🏭 Embedding 调用频率：仓库 `embeddings.py` 的本地 JSON 缓存就是一个真实的省钱机制——同一段文本永远不会被重复计费。
- 🏭 Agent Loop 次数：Agent/LangGraph 循环没有上限会导致 token 消耗失控，务必设 `recursion_limit` 或业务层重试上限。

### Performance
- 🏭 Streaming：`.stream()` 让用户第一时间看到逐字输出，而不是等模型把整段话生成完，仓库目前完全没有用到（都是 `.invoke()`）。
- 🏭 并行执行：`RunnableParallel`（如 Example 4 里 `context` 和 `question` 两路）能让互不依赖的步骤同时跑，而不是排队等待。
- 🏭 缓存：Embedding 缓存已经在仓库里实现；生产系统里模型响应本身也可以按"相同 Prompt 命中缓存"做类似优化。

### Security
- 🏭 Tool 权限：任何会执行"写"操作（发邮件、下单、删数据）的 Tool，都应该有额外的确认/权限校验，不能让模型单方面决定执行——这正是 LangGraph Human-in-the-loop 存在的意义。
- 🏭 Prompt Injection：如果 RAG 的检索内容来自用户可控的数据源（比如用户上传的文档），检索出来的文本本身可能包含"伪装成指令"的恶意内容，Prompt 里应该明确区分"系统指令"和"检索到的数据"，不能无脑信任检索结果。
- 🏭 敏感数据：仓库 `config.py` 通过 `.env`（已在 `.gitignore` 中）管理凭证，绝不硬编码密钥——这是最基本、也最容易被忽视的一条。
- 🏭 外部 API 访问：`documents.py` 的 `load_web_content()` 会对用户传入的任意 URL 发起请求，生产环境要防范 SSRF（比如禁止访问内网地址段）。

### Observability
- 🏭 Logging：目前仓库全部用 `print()`，生产系统应该换成结构化日志（包含 request_id、model_id、耗时、token 用量）。
- 🏭 Tracing/LangSmith：LangChain 生态提供 LangSmith 做端到端调用链路追踪（尤其是 Agent/Graph 这种多步骤流程，光靠日志很难还原完整决策路径）。
- 🏭 Token 追踪：结合 `usage_metadata` 做用量报表，是控制成本失控的第一道防线。

---

<a id="20"></a>
## 20. 学习路线（Level 1–6）

### Level 1 – 数据基础（仓库已完整覆盖，从这里开始）
- 目标：吃透 `Document` / Text Splitter 的设计意图。
- 必须掌握：`Document` 结构、Lazy Loading、`CharacterTextSplitter` vs `RecursiveCharacterTextSplitter`、`chunk_size`/`chunk_overlap`。
- 推荐练习：跑通 `documents.py`、`text_splitter.py`；试着换一份你自己的长文本，观察 `chunk_overlap=0` 和 `chunk_overlap=50` 检索结果的差异。
- 学完后的能力：能独立设计"如何把一批杂乱数据变成结构化 `Document`"的方案。

### Level 2 – 向量与检索（仓库已完整覆盖）
- 目标：理解 Embedding → Vector Store → Retriever 的完整链路，以及 Similarity vs MMR 的取舍。
- 必须掌握：`embed_query` vs `embed_documents`、`Chroma.from_documents` vs 直接构造读盘、`as_retriever()`、`similarity_search_with_score`。
- 推荐练习：跑通 `embeddings_chroma.py`、`embedding_chroma_retrieval.py`、`embedding_chroma_persistence.py`（记得跑两次观察持久化）；给示例数据加几篇内容重复的文档，对比 MMR 和纯相似度结果的差异。
- 学完后的能力：能给任意一批文档搭出一个可用的语义检索系统。

### Level 3 – Model / Prompt / Runnable 基础（仓库部分覆盖）
- 目标：把"裸调用模型"升级成"用 Prompt Template + LCEL 组合"。
- 必须掌握：`ChatPromptTemplate`、`|` / `RunnableSequence`、`RunnableParallel`、`StrOutputParser`。
- 推荐练习：把 `check_connection.py` 改写成 `prompt | llm | StrOutputParser()` 的链式写法。
- 学完后的能力：能写出可复用、可组合的 LCEL Chain，而不是散落的手写调用代码。

### Level 4 – 补齐 RAG 生成端（仓库预留了缺口，本总结第 14 节 Example 4 已给出实现）
- 目标：把 Level 1–3 的所有组件拼成一条完整 RAG 链，正式补完 `rag_pipeline.py`。
- 必须掌握：`RunnableParallel` 组装 context+question、`format_docs`、防止模型"绕过检索结果自由发挥"的 Prompt 约束写法。
- 推荐练习：按第 14 节 Example 4 的代码实现 `rag_pipeline.py`，再故意问一个知识库里没有的问题，验证模型是否老实说"不知道"。
- 学完后的能力：能独立搭建一条端到端可用的 RAG 问答系统。

### Level 5 – Tools / Agents（仓库完全未涉及，属于下一阶段）
- 目标：让系统从"只能回答"进化到"能做事、能自主决策"。
- 必须掌握：`@tool`、`bind_tools`、Tool Calling 循环、`create_agent`。
- 推荐练习：按第 14 节 Example 3 手写一次 Tool Calling 循环（不用 LangGraph），体会"如果不用 LangGraph，这个循环要自己维护多少状态"。
- 学完后的能力：能设计并实现一个具备外部工具调用能力的 Agent。

### Level 6 – LangGraph 与生产化（仓库完全未涉及，需要系统学习）
- 目标：掌握显式状态机编排，具备设计"可控、可持久化、可人工介入"的复杂工作流的能力。
- 必须掌握：`StateGraph`、`State`/reducer、`add_conditional_edges`、`checkpointer`、Human-in-the-loop。
- 推荐练习：按第 14 节 Example 5，把 Level 4 的 RAG 链改造成"检索质量不够就自动重写问题重试"的 LangGraph 版本；再加一个 `checkpointer`，中断程序后验证能否从断点恢复。
- 学完后的能力：能不依赖任何示例代码，从零设计并实现一个生产级 LangChain / LangGraph Agent 应用——这正是本总结开头设定的最终目标。

---

<a id="21"></a>
## 21. Cheat Sheet

### Core

```text
Chat Model
    ↓
model.invoke(messages)
Prompt
    ↓
ChatPromptTemplate.from_messages([...])
Chain
    ↓
prompt | model | parser
```

### Tool / Agent

```text
Tool
    ↓
@tool
Bind
    ↓
model.bind_tools([tool1, tool2])
Agent（高层）
    ↓
create_agent(model=..., tools=[...], system_prompt=...)
Agent（底层，需要自定义控制流时）
    ↓
StateGraph + State + Nodes + Edges
```

### RAG

```text
Documents → Text Splitter → Embeddings → Vector Store → Retriever → format_docs → Prompt → LLM → Answer
```

### LangGraph

```text
State（TypedDict） + Nodes（函数） + Edges/Conditional Edges + START/END
    ↓
builder.compile() → graph.invoke(initial_state)
```

### Execution

```text
.invoke()   单次同步调用
.batch()    批量调用
.stream()   流式输出
.ainvoke()  异步单次调用
```

### 常用 API 速查表

| Concept | API / Class | 用途 |
|---|---|---|
| Chat Model | `ChatBedrockConverse`（或任意 `BaseChatModel` 实现） | 对话式模型调用 |
| Message | `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` | 带 role 的消息 |
| Prompt | `ChatPromptTemplate` | 变量化、多角色提示词模板 |
| Runnable 组合 | `\|`（`RunnableSequence`）、`RunnableParallel`、`RunnableLambda`、`RunnablePassthrough` | 组件组合 |
| Output Parser | `StrOutputParser`、`with_structured_output(Schema)` | 把模型输出转成结构化结果 |
| Document | `langchain_core.documents.Document` | 文本 + metadata 容器 |
| Text Splitter | `CharacterTextSplitter`、`RecursiveCharacterTextSplitter` | 文本分块 |
| Embeddings | `BedrockEmbeddings`（或任意 `Embeddings` 实现） | 文本→向量 |
| Vector Store | `Chroma`（`from_documents` / 构造函数读盘） | 存储向量、相似度检索 |
| Retriever | `vector_store.as_retriever(search_type=..., search_kwargs=...)` | 标准化检索接口 |
| Tool | `@tool` | 把函数暴露给模型调用 |
| Agent（高层） | `langchain.agents.create_agent` | 快速搭建 Tool-calling Agent |
| State | `TypedDict` / `Annotated[list, add_messages]` | 图执行过程中的共享数据 |
| Node | 普通函数 `(state) -> dict` | 图上的一步 |
| Edge | `add_edge` / `add_conditional_edges` | 节点间的固定/条件流转 |
| Graph | `StateGraph(...).compile()` | 构建并编译可执行的图 |
| Persistence | `checkpointer`（如 `InMemorySaver`） | 状态持久化，支持断点续跑 |

---

<a id="22"></a>
## 22. Version / API 变化说明 ⚠️

- **LangChain / LangGraph 均已发布 1.0**：官方博客明确了新的推荐路径——用 `langchain.agents.create_agent` 构建 Agent，它内部构建在 LangGraph 之上。旧的 `AgentExecutor` + `initialize_agent`（更早期）、以及后来的 `langgraph.prebuilt.create_react_agent`，现在都被视为 **legacy / deprecated** 路径，新项目不应该再以它们为起点（来源：[LangChain and LangGraph Agent Frameworks Reach v1.0 Milestones](https://www.langchain.com/blog/langchain-langgraph-1dot0)、[Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents)）。
- **Middleware 概念**：1.0 引入了"中间件"机制，用于在 Agent 循环的各个节点插入自定义逻辑（人工审批、消息摘要、PII 脱敏等），是对"直接改 `StateGraph` 节点"这种更底层做法的高层封装。
- **`langchain-community` 处于 sunset 状态**：仓库自身的 `CLAUDE.md`/`documents.py` 已经明确指出这一点，并给出了替代方案（直接用底层库构造 `Document`，或使用 `langchain-chroma`、`langchain-aws` 这类各自独立维护的官方集成包）。`pyproject.toml` 里仍然存在的 `langchain-community` 依赖，只是被其它包间接拉入，不代表仓库代码本身在用它。
- **导入路径**：`Chroma` 应从 `langchain_chroma` 导入（仓库写法正确），而不是更早期、已废弃的 `langchain.vectorstores.Chroma`。
- 由于这些 API 处于快速演进期，实际动手写代码前，建议直接核对当前安装的 `langchain` / `langgraph` / `langchain-aws` / `langchain-chroma` 具体版本号（`uv.lock` 里能查到），并以对应版本的官方文档为准——本节内容基于 2026 年 9 月的公开信息整理，具体参数名/默认值仍可能随小版本更新变化。

---

<a id="23"></a>
## 23. 自测题

完成本总结后，你应该能不查资料回答以下问题；如果某一题卡住了，回到对应章节复习。

### Beginner
1. `Document` 有哪两个字段？为什么只有其中一个参与 Embedding 计算？
2. `CharacterTextSplitter` 和 `RecursiveCharacterTextSplitter` 的核心区别是什么？仓库哪个示例函数直接展示了这个区别？
3. `embed_query` 和 `embed_documents` 为什么是两个不同的方法？
4. `similarity_search` 和 `similarity_search_with_score` 分别在什么场景下使用？
5. Chat Model 和普通 LLM 接口的本质区别是什么？
6. `.invoke()`、`.batch()`、`.stream()` 分别解决什么问题？
7. 为什么 `ChatPromptTemplate` 比 f-string 拼接字符串更适合正式项目？
8. `chunk_overlap` 存在的意义是什么？
9. Vector Store 里的 `metadata` 起什么作用？
10. 仓库为什么放弃使用 `langchain_community` 的 Loader？

### Intermediate
11. `\|` 运算符在 LangChain 里具体做了什么？
12. `RunnableParallel` 和 `RunnableSequence` 的区别？各自适合什么场景？
13. Tool Calling 的完整流程有哪几步？`ToolMessage` 里的 `tool_call_id` 为什么必须对应正确？
14. Agent 为什么会"循环"？循环什么时候停止？
15. Similarity Search 和 MMR 检索在算法上的核心差异是什么？为什么仓库的示例里两者结果一样？
16. `as_retriever()` 相比直接调用 `vector_store.similarity_search()` 有什么架构上的优势？
17. `embedding_chroma_persistence.py` 是怎么判断"该重新 Embedding 建库"还是"直接读盘"的？
18. 为什么 Chroma 在 Windows 上不能直接用 `tempfile.TemporaryDirectory()` 的自动清理？
19. Memory 和 State 有什么区别？
20. Structured Output 为什么比"在 Prompt 里要求模型输出 JSON"更可靠？

### Advanced
21. 如果要把 `rag_pipeline.py` 补完，需要用到本总结哪几节的哪些组件？完整数据流是怎样的？
22. LangGraph 的 `State` 里，普通字段和带 `add_messages` reducer 的字段在"合并新值"时的行为区别是什么？
23. 为什么说"LangChain 1.0 的 `create_agent` 本质上是预先搭好的 LangGraph 图"？这句话具体指什么？
24. 设计一个"检索结果不够好就自动重写问题重试"的 LangGraph 流程，画出它的图结构，并说明哪个字段负责防止死循环。
25. Human-in-the-loop 在 LangGraph 里是怎么实现"暂停等待人工输入"的？这依赖哪个底层能力（提示：和 Persistence 是同一套机制）？
26. 为什么 Tool 权限控制、Prompt Injection 防范在生产环境里比在本地开发时更重要？结合仓库 `load_web_content()` 这个具体函数说明可能的风险点。
27. 如果要把仓库现在的 Bedrock 换成另一家模型供应商，哪些文件完全不用改？哪些需要改？为什么？
28. `RunnableLambda` 在什么场景下是必需的，而不能直接把一个普通函数塞进 `\|` 链条？
29. LangSmith / Tracing 对调试 Agent Loop 的价值，和第 18 节提到的"打印 State 快照"相比，分别在什么情况下更有效？
30. 尝试不参考本总结，从零列出一个"生产级 RAG Agent"需要具备的全部组件清单，并标注每个组件对应 LangChain/LangGraph 里的哪个概念。
