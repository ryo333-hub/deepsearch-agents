"""
Markdown 转 PDF 工具

供主智能体把已经生成的 Markdown 文档转换为 PDF。Tool 层只负责解析当前
会话目录中的输入/输出路径，真正的版式转换交给 app.utils.word_converter。
"""

import logging
from pathlib import Path

try:
    from typing import Annotated, Optional
except ImportError:
    from typing_extensions import Annotated, Optional

from langchain_core.tools import tool

from app.api.context import get_session_context
from app.api.monitor import monitor
from app.utils.path_utils import PATH_ACCESS_DENIED_MESSAGE, resolve_path
from app.utils.word_converter import convert_md_to_pdf as convert_md_to_pdf_via_word


@tool
def convert_md_to_pdf(
    md_filename: Annotated[
        str,
        "当前 session 内 Markdown 相对路径，例如 reports/report.md；禁止绝对路径、../、..\\、盘符或 UNC 路径。",
    ],
    pdf_filename: Annotated[
        Optional[str],
        "当前 session 内 PDF 相对输出路径，例如 reports/report.pdf；可选，默认与源文件同目录同名。禁止绝对路径和 traversal。",
    ] = None,
) -> str:
    """
    使用当前 session 内相对路径，将 Markdown 文档转换为 PDF。

    :param md_filename: 当前 session 内 Markdown 相对路径，缺少后缀时会自动补为 .md
    :param pdf_filename: 当前 session 内可选 PDF 相对输出路径；不传时与 Markdown 同目录同名
    :return: 转换结果说明
    """
    monitor.report_tool("Markdown转PDF工具")

    try:
        # 输入和输出分别通过同一个 session 边界，Guard 先于 exists/open/PDF 转换。
        session_dir = get_session_context()

        # 先验证未经变换的用户输入，防止 with_suffix 等操作把 ADS/特殊路径
        # 改写成看似安全的普通文件名。
        raw_md_abs_path = Path(resolve_path(md_filename, session_dir))
        session_path = Path(session_dir).resolve(strict=False)
        md_relative_path = raw_md_abs_path.relative_to(session_path).with_suffix(".md")
        md_abs_path = Path(resolve_path(md_relative_path.as_posix(), session_dir))

        # 未指定 PDF 文件名时，默认与源 Markdown 同目录同名
        if pdf_filename:
            raw_pdf_abs_path = Path(resolve_path(pdf_filename, session_dir))
            pdf_relative_path = raw_pdf_abs_path.relative_to(session_path).with_suffix(
                ".pdf"
            )
        else:
            pdf_relative_path = md_relative_path.with_suffix(".pdf")
        pdf_abs_path = Path(resolve_path(pdf_relative_path.as_posix(), session_dir))

        if not md_abs_path.exists():
            return "错误：Markdown 文件不存在。"

        # PDF 版式、中文字体和 Markdown 解析细节都封装在底层转换模块中。
        # 底层只处理内部绝对路径；返回给模型时仅暴露 session-relative path。
        conversion_result = convert_md_to_pdf_via_word(md_abs_path, pdf_abs_path)
        if conversion_result.startswith("成功转换:") and pdf_abs_path.exists():
            return f"成功转换: {pdf_relative_path.as_posix()}"
        if conversion_result == "缺少依赖库，请安装 reportlab":
            return conversion_result
        return "转换失败：PDF 文件未生成。"

    except ValueError:
        return PATH_ACCESS_DENIED_MESSAGE
    except Exception as e:
        logging.error(f"转换失败: {e}", exc_info=True)
        return "转换失败：PDF 文件未生成。"


if __name__ == "__main__":
    # 本地调试入口：直接运行本文件可验证 Markdown 转 PDF 链路
    get_session_context = lambda: "./examples/test_docs"

    test_dir = Path("./examples/test_docs/sub_dir")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_md_path = test_dir / "金融电商行业分析报告.md"
    test_md_path.write_text(
        """# 金融电商行业分析报告

## 一、核心结论

金融与电商业务正在加速融合，平台需要同时关注用户增长、交易转化、支付体验和风险控制。
在制定 2026 年业务策略时，应优先分析消费金融场景、直播电商转化效率和用户资金安全。

## 二、重点观察

- 电商平台的增长重点从单纯拉新转向精细化运营和复购提升。
- 支付、分期和信用评估能力会直接影响高客单价商品的转化率。
- 金融业务接入电商场景时，需要同步关注合规披露、风控策略和用户体验。

## 三、示例数据

| 指标 | 观察结果 | 建议动作 |
| --- | --- | --- |
| 支付转化 | 高客单价订单更依赖便捷支付 | 优化分期和组合支付入口 |
| 用户复购 | 会员权益影响长期价值 | 结合金融权益设计会员激励 |
| 风险控制 | 异常交易和欺诈风险上升 | 强化实时风控和交易监测 |

## 四、行动建议

围绕“增长、转化、风控、合规”四个关键词设计分析框架，并将公开信息、数据库数据和 RAGFlow 知识库材料统一整理成可交付报告。
""",
        encoding="utf-8",
    )

    print(convert_md_to_pdf.invoke({"md_filename": "sub_dir/金融电商行业分析报告.md"}))
