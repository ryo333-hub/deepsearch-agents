# 最终架构与代码入口

冻结状态：Core integration capabilities validated；final synthesis reliability not fully validated。架构图见[根 README](../README.md#architecture)。本页描述现有代码，不是拟建方案。

## 请求与回传

1. [前端 session hook](../frontend/src/hooks/useDeepAgentSession.ts) 管理任务与事件。[API 客户端](../frontend/src/lib/api.ts) 请求当前 `thread_id` 的 KB 列表，将选择的 `knowledge_base_id` 连同 query 提交。
2. [FastAPI server](../app/api/server.py) 接收 `POST /api/task`，校验输入与 KB 所属 session，创建异步任务；WebSocket `/ws/{thread_id}` 按会话接收事件。
3. [run_deep_agent](../app/agent/main_agent.py) 再次校验所选 KB，准备 workspace，将上传附件复制进去；设置三个 ContextVar，给 Main 注入合法 KB ID 和相对文件路径说明。
4. 同文件中的 `create_deep_agent` 注册三个业务助手与 `InMemorySaver`。Main 自主使用 `task` 委派子任务，description 传递检索问题与 KB ID；返回的工具消息进入 Main 的综合上下文。并非把完整聊天记录无条件复制给每个助手。
5. Main 的最终内容经 [monitor](../app/api/monitor.py) 发送 `task_result`，前端展示答案、Citation 文本与 URL。工具事件也经 monitor 推送。`astream` 消费图状态，当前 UI 不是逐 token 答案流。

## 三类业务助手

| 注册配置 | 工具 | 实际数据链路 |
| --- | --- | --- |
| [Database](../app/agent/subagents/database_query_agent.py) | list_sql_tables / get_table_data / execute_sql_query | MySQL connector → 电商库，只读账号与 SQL 防护 |
| [Local Knowledge](../app/agent/subagents/knowledge_base_agent.py) | search_local_knowledge_base | session/KB 校验 → E5 query embedding → NumPy 检索 → 片段与 Citation |
| [Web](../app/agent/subagents/network_search_agent.py) | internet_search | Tavily → 公开摘要与来源 URL |

当前 `openai:deepseek-flash` harness profile 关闭默认 general-purpose subagent。Main 还拥有 `read_file_content`、`generate_markdown`、`convert_md_to_pdf`；未增加 Redis、MCP、持久 checkpoint 或额外业务助手。

## 上下文与状态

| 字段/组件 | 生命周期与用途 |
| --- | --- |
| thread_id | 前端选择的会话标识；同时作为 checkpoint key、WebSocket 定向标识和 session 目录标识 |
| knowledge_base_id | 用户显式选择；API/运行入口及 Knowledge Tool 复核所属 session，不默认猜测 KB |
| ContextVar | [context.py](../app/api/context.py) 保存 thread、selected KB、session_dir；异步工具读取，任务结束 finally 恢复 |
| InMemorySaver | 对话图状态，仅当前后端进程内有效；重启丢失 |
| session workspace | 上传暂存 `app/updated/session_{thread_id}`，执行文件 `app/output/session_{thread_id}`；路径检查限制跨 session 访问 |
| KB storage | `.data/local_rag/` 下按 session/KB 保存 manifest、文档片段、NumPy 向量 generation 与 current 指针；不依赖聊天 checkpoint |
| Citation | 检索证据含文档/片段 ID、来源和位置；Knowledge 返回 `[C…]`，Main 保留引用；不等于程序自动证明综合陈述正确 |

## Local RAG

[prepare_ecommerce_demo_kb.py](../scripts/prepare_ecommerce_demo_kb.py) 用同一库存 SOP 建立会话 KB：加载 → 结构与 token 预算分块 → CPU embedding → 向量索引。相同 session/内容且索引完整时复用，不覆盖历史 KB。

[config.py](../app/local_rag/config.py) 固定 `intfloat/multilingual-e5-small` revision，384 维、512 token 上限；文档使用 `passage:`，查询使用 `query:`。运行时仅加载预备好的本地文件。[vector_store.py](../app/local_rag/vector_store.py) 使用 NumPy 文件持久化向量；[retrieval.py](../app/local_rag/retrieval.py) 对归一化向量做内积检索。没有外部向量数据库。

## 可信边界

数据库事实、知识规则、公开背景在最终回答中应分开。第三轮真实执行成功，但 Main 额外断言单仓覆盖不足 0.5 天，工具并未返回分仓销量。这是 synthesis 证据约束不足，不能由架构链路成功推导回答完全可靠。

当前 session 防护是本地 Demo 隔离，不替代生产身份认证；单进程状态与文件存储也不代表多节点部署能力。
