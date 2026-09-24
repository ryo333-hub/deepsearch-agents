# DeepSearch Agents Project Freeze

冻结日期：2026-09-24。停止功能开发与真实模型复测；本阶段仅文档、已有截图整理及 Git 忽略规则，无 commit / push。

## Project status

**Core integration capabilities validated** 与 **final synthesis reliability not fully validated** 同时成立。608/608 离线 PASS；三次真实 Main 验收均保留 FAIL；Database 跨月付款仍为数据覆盖缺口 `blocked_preflight`。

## Architecture / README changes

根 README 已按真实代码整理项目定位、能力、依赖、Mermaid 架构图、Demo、从新环境准备的启动命令、测试说明及 Known Limitations。详细调用链见 [architecture.md](architecture.md)。当前 Knowledge 使用本地 RAG，RAGFlow 是保留的教学代码，不是当前 Main 的注册助手。

## Demo structure

保留 `demo/ecommerce/` 的 scenario、知识文档源、SOP、确定性验收和全部历史真实 JSON；通过目录 README 建立索引，不搬迁或覆盖历史结果。`.data/ecommerce/` 的密码与 `.data/local_rag/` 的模型/索引属于本地运行状态。

## Screenshots selected

| 文件 | 来源与用途 |
| --- | --- |
| [ecommerce-demo-home.png](images/ecommerce-demo-home.png) | 原 `.tmp/main_fidelity_before.png` 的逐字节副本；真实第三轮开始前首页，展示 KB selector、三类助手和任务入口；根 README 主图 |
| [第三轮最终页面](../demo/ecommerce/main_agent_demo_clean_retest_3_page.png) | 已有失败证据，保留在验收目录；含无证据单仓指标，不作为成功 Hero Demo |
| docs/images/deepsearch-*.jpg / .svg | 原教学展示素材，保留但不作为本轮电商或 Local RAG 的验证证据 |

现有 `.tmp/main_demo_after.png`、`main_synthesis_after.png`、`main_fidelity_after.png` 均属于失败轮次。没有把局部正确截图包装成完整成功回答；未重新调用模型，也未伪造新的检索过程截图。Database SQL、Knowledge Citation 与三源调用过程可直接查验第三轮 JSON 的 `tools` / `llm` / `monitor`。

## Test status / Real LLM acceptance status

608 项离线结果取自冻结前的 [报告](../demo/ecommerce/main_fact_fidelity_offline_results.json)，本次文档整理不冒充新一轮回归。第三轮真实环境使用 deepseek-flash：DeepSeek 23 次、Tavily 4 次、工具 37 次、320,150 tokens。当前整理阶段新增模型请求为 0。

分类保真、精确日期窗口、跨仓覆盖、三源自主路由、引用传递和页面显示均通过；建议中的单仓覆盖仍缺少证据。未经第四轮修复，不标记整体 validated。

## Known limitations

- InMemorySaver 重启丢失聊天 checkpoint；磁盘 KB/文件独立保存。
- 本地单进程原型，不是多节点生产系统，也没有生产身份认证。
- 无跨月付款样本，该边界尚未真实 LLM 验证。
- 最终综合可能额外生成缺乏工具支持的派生指标。
- Web 只能作为背景，不自动解释内部业务或建立商品分类映射。

## Git ignore / secret check

`.gitignore` 保留 `.env.example`，忽略 `.env` / `*.env`、`.data/`、`.tmp/`、`.tools/`、会话输出与上传、node_modules、虚拟环境、Python 缓存、前端构建目录和本地缓存。模型快照位于 `.data/`，不提交。

打包检查针对 Git 当前候选文件做本地凭据精确匹配及常见 token 格式扫描，只输出文件名/计数，不输出 secret。它不是 Git 全历史安全审计；不要使用 `git add -f` 绕过忽略规则。

本轮检查结果：178 个候选文件中，本地 API/数据库凭据精确匹配为 0，常见 token/私钥格式命中为 0；没有已跟踪的 `.env`、`.data/`、临时日志或 session 文件。文档本地链接均存在，首页截图与原文件哈希一致。冻结基线内仅 Demo README 发生变化，运行代码、Prompt、测试、脚本、schema、SOP 和所有验收历史均未改变。未执行依赖下载、数据库初始化或真实 LLM 复测。

## Files changed

本阶段改动：根 README、`.gitignore`、本说明、架构说明、Demo README，以及一张已有首页截图副本。未修改运行代码、Prompt、测试、Harness、schema、数据源或历史验收结论。

## Files recommended for commit

- `README.md`、`docs/architecture.md`、`docs/project-freeze.md`、`docs/images/ecommerce-demo-home.png`、`.gitignore`。
- 经本地扫描的 `app/`、`frontend/` 源码及前端依赖清单/锁文件。
- `scripts/`、`tests/`、`demo/ecommerce/` 的数据说明、SOP、验收报告与失败截图。
- `docker/`、`.env.example`、`pyproject.toml`、`uv.lock`；保留已有 `requirements.txt`，当前环境以 uv 锁文件为准。

运行代码等包含之前各阶段尚未提交的改动，并非本次文档整理新增；提交时应一并审阅。本阶段未暂存、commit 或 push。

## Files that must not be committed

`.env`、`.data/ecommerce/demo.env`、`ecommerce_ro.json`、API/数据库密码、`.tmp/` 的原始日志和临时观察入口、`.tools/`、`.data/local_rag/` 的模型及 session KB、`app/output/`、`app/updated/`、node_modules、虚拟环境与各类缓存。
