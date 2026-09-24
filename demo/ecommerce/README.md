# 电商 Main Agent 三源 Demo

**功能冻结：Core integration capabilities validated；final synthesis reliability not fully validated。**
三次真实 Main 验收均为 FAIL；保留失败证据，不再自动修复或调用真实模型。

## 目录与数据说明

| 文件 | 用途 |
| --- | --- |
| `scenario.json` | 合成电商场景与日期/业务口径；不是企业真实数据 |
| `knowledge_docs_source.json` / `knowledge_base/` | 演示内部资料与四份 Markdown 文档；Main Demo 使用 `02_库存经营SOP.md` |
| `acceptance_results.json` | 确定性数据验收 |
| `knowledge_acceptance_results.json` / `network_acceptance_results.json` | 知识与网络链路历史验收 |
| `real_llm_acceptance_results.json` | 汇总历史；保留 Database `blocked_preflight` 和三轮 Main 记录 |
| `main_agent_demo_integration_results.json` | 第一次三源真实验收 FAIL，原始 Trace |
| `main_agent_demo_clean_retest_2.json` | 第二次 FAIL：无依据商品分类改写 |
| `main_agent_demo_clean_retest_3.json` | 第三次 FAIL：无分仓销量却推导单仓覆盖；包含原始 Trace、Ground Truth 和复核 |
| `*_final*.md` / `*_final_answer.md` | 原始模型回答，不是经过人工修正的业务报告 |
| `main_agent_demo_clean_retest_3_page.png` | 第三次最终页面失败证据，不作为成功展示图 |
| `main_fact_fidelity_offline_results.json` | 冻结前 608/608 离线 PASS；保留其他历史离线报告 |

数据库生成器为 [seed_ecommerce_demo.py](../../scripts/seed_ecommerce_demo.py)，schema 为 [ecommerce_schema.sql](../../docker/mysql/ecommerce_schema.sql)。交易数据覆盖 2026 年 6—8 月，库存为 2026-09-01 快照。有效成交以 paid_at 和 paid/completed 统计；自然月销量与 SOP 最近 30 天窗口不是同一口径。

首次环境配置、Docker、只读账号及模型下载步骤见[根 README Quick Start](../../README.md#quick-start)。下面命令适用于已准备好 MySQL 与本地模型的环境，本次打包未重新执行初始化或验收。

使用真实 Main 自主路由 Database、企业知识和网络搜索，不预设子 Agent 调用顺序。
数据库和库存 SOP 均为虚构演示资料。现有通用 `.env` 与默认数据库用途不变。

## 准备与启动（仓库根目录，PowerShell）

前置：现有 MySQL demo 已部署到 localhost:3307，具有 ecommerce_ro 只读账号；
`.env` 已配置 DeepSeek 和 Tavily；现有 Local RAG 的 multilingual-e5-small 固定版本模型已在本地。
准备脚本不下载 embedding 模型，不调用 LLM。

```powershell
# 一次性从已有本地只读凭据生成持久 Demo profile；重复执行保留现有配置。
.\.venv\Scripts\python.exe scripts/start_ecommerce_demo.py --init-config
.\.venv\Scripts\python.exe scripts/start_ecommerce_demo.py --check

# 生成全新 session、准备 SOP 知识库，并输出 KB ID 和页面链接。
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_ecommerce_demo_kb.py

# 后端固定加载 .data/ecommerce/demo.env，无需手工临时覆盖 MYSQL_*。
.\.venv\Scripts\python.exe scripts/start_ecommerce_demo.py

# 另一个终端
cd frontend
pnpm install --frozen-lockfile
npm run dev
```

配置文件 `.data/ecommerce/demo.env` 和只读凭据文件均被 Git 忽略，不提交 secret。
若没有自动生成用的凭据文件，可手工创建该 profile，填入 MYSQL_HOST、MYSQL_PORT、
MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE；仍须使用 ecommerce_ro / insight_ecommerce_db。

打开准备脚本输出的 `http://localhost:5173/?thread_id=...`，确认知识库选择器显示
“电商库存 SOP”。URL 只选择 session，不在 Prompt 中写死 KB ID。
普通页面上的“新建研搜”会生成新 session，不能访问旧 session 的知识库。
可使用页面 THREAD 的完整 ID（悬停显示）重新准备：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_ecommerce_demo_kb.py --thread-id <当前thread_id>
```

然后点击“刷新知识库”。相同 session、相同 SOP 内容且索引完整时复用 KB。
文档变更会创建新 KB，不覆盖旧记录；要完整重置演示上下文，省略 --thread-id 创建新 session。
准备文件使用独立临时目录，不把 SOP 污染为 Main 的上传附件。

## 页面链路

`GET /api/knowledge-bases?thread_id=...` 只读返回当前 session 的 KB 摘要。
选择器将真实 knowledge_base_id 传入 POST /api/task，再由现有 run_deep_agent、
ContextVar 和 Knowledge Tool 校验与使用。普通上传只保存附件，不自动建立知识库。

点击三源综合示例，或输入：

> 分析 2026 年 8 月销量最高的商品及当前库存情况，依据公司的库存 SOP 判断缺货风险；再检索近期公开的美妆消费趋势作为背景，给出简短建议，并分别标明数据库依据、内部文档引用和公开来源链接。

最终回答应区分数据库事实、内部 SOP 判断及外部背景，保留内部 Citation 和公开 URL。
库存是 demo 的 2026-09-01 快照，不能当作当天实时库存。
网络趋势不能直接证明内部销量或库存变化的原因。

## 边界

- 当前无跨月付款样本，属于数据覆盖缺口；跨月 paid_at 尚未真实 LLM 验证。
- InMemorySaver 重启不保留对话；本地 KB 与索引保留。
- 当前适用于本地单进程演示，session ID 不是生产用户认证机制。
- 不自动启动模型验收；每次点击提交都会发生真实模型请求。
- 第三轮保留“个护”原始分类、补查精确日期窗口并正确计算跨仓覆盖，但额外单仓指标没有分仓销量支持。工具链已运行不等于所有综合陈述可靠。
- 公开 Web 信息是背景，不能直接推导具体 SKU 的需求或缺货风险变化。
