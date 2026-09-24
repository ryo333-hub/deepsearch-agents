"""
MySQL 数据库查询工具模块

封装数据库查询助手使用的三个 LangChain 工具：
list_sql_tables 用于发现真实表名，get_table_data 用于预览字段和样例数据，
execute_sql_query 用于在确认结构后执行自定义查询。
"""

import os
import re

from dotenv import load_dotenv
from langchain_core.tools import tool
from mysql.connector import Error, connect

from app.api.monitor import monitor

load_dotenv()


_READONLY_STATEMENTS = frozenset({"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"})
_TABLE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TABLE_EXISTS_QUERY = (
    "SELECT 1 FROM information_schema.tables "
    "WHERE table_schema = %s AND table_name = %s AND table_type = 'BASE TABLE' "
    "LIMIT 1"
)
_STATEMENT_KEYWORDS = frozenset(
    {
        *_READONLY_STATEMENTS,
        "INSERT",
        "UPDATE",
        "DELETE",
        "REPLACE",
        "CREATE",
        "ALTER",
        "DROP",
        "TRUNCATE",
        "RENAME",
        "GRANT",
        "REVOKE",
        "CALL",
        "DO",
        "SET",
        "LOCK",
        "UNLOCK",
        "LOAD",
        "ANALYZE",
        "OPTIMIZE",
        "REPAIR",
    }
)


def _tokenize_sql(query: str) -> list[str]:
    """提取字符串和普通注释之外的 SQL 词元，用于最小只读校验。"""
    tokens = []
    current = []
    parenthesis_depth = 0
    index = 0

    def flush_current():
        if current:
            tokens.append("".join(current).upper())
            current.clear()

    while index < len(query):
        char = query[index]

        # MySQL 的 /*! ... */ 会执行注释体，不能作为普通注释忽略。
        if query.startswith("/*", index):
            flush_current()
            if query.startswith("/*!", index):
                raise ValueError("禁止 MySQL 可执行注释")
            comment_end = query.find("*/", index + 2)
            if comment_end == -1:
                raise ValueError("SQL 包含未闭合的块注释")
            index = comment_end + 2
            continue

        # MySQL 仅在 -- 后是空白或行尾时将其识别为行注释。
        if query.startswith("--", index) and (
            index + 2 == len(query) or query[index + 2].isspace()
        ):
            flush_current()
            newline = query.find("\n", index + 2)
            index = len(query) if newline == -1 else newline + 1
            continue

        if char == "#":
            flush_current()
            newline = query.find("\n", index + 1)
            index = len(query) if newline == -1 else newline + 1
            continue

        # 字符串和反引号标识符中的分号、关键字不参与语句边界判断。
        if char in {"'", '"', "`"}:
            flush_current()
            quote = char
            index += 1
            while index < len(query):
                if query[index] == "\\":
                    index += 2
                    continue
                if query[index] == quote:
                    if index + 1 < len(query) and query[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                raise ValueError("SQL 包含未闭合的引号")
            continue

        if char.isalnum() or char in {"_", "$"}:
            current.append(char)
            index += 1
            continue

        flush_current()
        if char == ";":
            tokens.append(char)
        elif char == "(":
            parenthesis_depth += 1
            tokens.append(char)
        elif char == ")":
            parenthesis_depth -= 1
            if parenthesis_depth < 0:
                raise ValueError("SQL 括号不匹配")
            tokens.append(char)
        index += 1

    flush_current()
    if parenthesis_depth != 0:
        raise ValueError("SQL 括号不匹配")
    return tokens


def _contains_keyword_sequence(tokens: list[str], *keywords: str) -> bool:
    """判断 SQL 词元中是否出现连续关键字，括号不作为关键字。"""
    words = [token for token in tokens if token not in {"(", ")", ";"}]
    sequence_length = len(keywords)
    return any(
        tuple(words[index : index + sequence_length]) == keywords
        for index in range(len(words) - sequence_length + 1)
    )


def _validate_readonly_query(query: str) -> None:
    """只允许单条只读 SQL；不满足要求时抛出带简短原因的 ValueError。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("SQL 不能为空")

    tokens = _tokenize_sql(query)
    if not tokens:
        raise ValueError("SQL 不能为空")

    semicolon_positions = [
        index for index, token in enumerate(tokens) if token == ";"
    ]
    if len(semicolon_positions) > 1 or (
        semicolon_positions and semicolon_positions[0] != len(tokens) - 1
    ):
        raise ValueError("禁止多语句")
    if semicolon_positions:
        tokens = tokens[:-1]
    if not tokens:
        raise ValueError("SQL 不能为空")

    first_keyword = tokens[0]
    if first_keyword == "WITH":
        depth = 0
        final_statement = None
        for token in tokens[1:]:
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
            elif depth == 0 and token in _STATEMENT_KEYWORDS:
                final_statement = token
                break
        if final_statement != "SELECT":
            raise ValueError("WITH 最终必须执行只读 SELECT")
    elif first_keyword not in _READONLY_STATEMENTS:
        raise ValueError(f"禁止非只读语句：{first_keyword}")

    if _contains_keyword_sequence(tokens, "INTO", "OUTFILE") or _contains_keyword_sequence(
        tokens, "INTO", "DUMPFILE"
    ):
        raise ValueError("禁止 INTO OUTFILE / INTO DUMPFILE")

    # SELECT 的锁定读会改变并发行为，不属于本工具允许的普通只读查询。
    if _contains_keyword_sequence(tokens, "FOR", "UPDATE") or _contains_keyword_sequence(
        tokens, "LOCK", "IN", "SHARE", "MODE"
    ):
        raise ValueError("禁止锁定读取")

    # 这些函数可能读取服务器文件、持有锁或造成显著阻塞，按高风险查询拒绝。
    high_risk_functions = {"LOAD_FILE", "GET_LOCK", "RELEASE_LOCK", "SLEEP", "BENCHMARK"}
    if high_risk_functions.intersection(tokens):
        raise ValueError("禁止高风险 SQL 函数")


def _validate_table_name(table_name: object) -> str:
    """只接受当前项目需要的简单 MySQL 表标识符。"""
    if not isinstance(table_name, str) or not _TABLE_NAME_PATTERN.fullmatch(
        table_name
    ):
        raise ValueError("非法表名")
    return table_name


# 集中读取数据库配置，后续三个工具都复用这份连接参数
def get_db_config():
    """
    从环境变量读取 MySQL 连接配置

    所有数据库工具都通过此函数拿到同一份连接参数，避免每个工具重复读取环境变量
    :return: mysql.connector.connect 可直接使用的连接参数
    """
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),
    }

    # 去掉未配置的可选项，避免把 None 传给 mysql.connector 造成连接参数异常
    config = {k: v for k, v in config.items() if v is not None}

    # user/password/database 是本教程工具能正常查询业务库的最小必要配置
    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config


@tool
def list_sql_tables() -> str:
    """
    查询当前数据库中所有可用表

    作用：让模型先识别真实可用的表名，方便后续预览表结构和编写自定义 SQL。
    :return: 有表：可用的表有：表1,表2,表3...
             没有表：没有可用的表
             出现异常：查询出现异常：异常信息
    """

    # 埋点：工具一被调用，前端可以展示当前正在查询数据库表名
    monitor.report_tool(tool_name="数据库表名查询工具：list_sql_tables", args={})

    # 加载数据库连接信息
    config = get_db_config()

    # MySQL 查询的固定步骤：
    # 1. 创建连接
    # 2. 创建 cursor
    # 3. 执行 SQL
    # 4. 获取返回结果
    # 5. 释放连接和 cursor 资源
    # 这里捕获异常并返回中文提示，避免工具报错直接中断 Agent 执行链路
    try:
        # 使用 with 管理连接和游标，查询结束后自动释放数据库资源
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                sql = "SHOW TABLES"
                cursor.execute(sql)

                # SHOW TABLES 返回形如：[("drugs",), ("inventory",), ("sales_records",)]
                tables = cursor.fetchall()
                if not tables:
                    return "没有可用的表"

                # 取每个元组的第一个元素，拼成模型容易阅读的表名列表
                table_names = [table[0] for table in tables]
                return f"可用的表有：{', '.join(table_names)}"
    except Error as e:
        return f"查询出现异常：{str(e)}"


@tool
def get_table_data(table_name) -> str:
    """
    查询指定表的前 100 行数据

    当前工具调用之前，应先调用 list_sql_tables 完成表名校验。
    此工具的作用：
    1. 完成单表样例数据查询
    2. 为多表查询提供表结构信息和数据格式参考
    :param table_name: 表名
    :return: CSV 格式数据
             1. 第一行是列信息，列之间使用英文逗号分隔
             2. 第二行开始是表数据，值之间也使用英文逗号分隔
             3. 行和行之间使用 \n 分隔
             4. 至多查询 100 条表数据
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n
                1,张三,18\n -> 至多查询 100 条
    """
    # 明显非法的标识符在读取配置和建立数据库连接之前直接拒绝。
    try:
        validated_table_name = _validate_table_name(table_name)
    except ValueError:
        return "拒绝执行：非法表名。"

    # 埋点：工具二被调用，前端可以展示当前正在预览哪张表
    monitor.report_tool(
        tool_name="数据库表数据查询工具：get_table_data",
        args={"table_name": validated_table_name},
    )

    # 获取数据库参数
    config = get_db_config()

    # 查询流程同样是：连接 -> cursor -> 执行 SQL -> 获取列信息和数据 -> 自动释放资源
    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                # 使用参数化元数据查询，只允许当前数据库中真实存在的普通表。
                cursor.execute(
                    _TABLE_EXISTS_QUERY,
                    (config["database"], validated_table_name),
                )
                if cursor.fetchone() is None:
                    return "表不存在或不允许访问。"

                # 表名已通过严格正则和真实 BASE TABLE 白名单，此处再用反引号引用。
                sql = f"SELECT * FROM `{validated_table_name}` LIMIT 100"
                cursor.execute(sql)

                # cursor.description 保存查询结果的列元信息
                # 例如：[("id", ...), ("name", ...), ("age", ...)]
                # 如果 SQL 没有结果集，description 可能为 None
                description = cursor.description
                if not description:
                    return f"数据表 {validated_table_name} 暂无数据。"

                # 只取每个列信息元组的第一个元素，也就是列名
                # 例如：["id", "name", "age"]
                columns = [desc[0] for desc in description]

                # fetchall 返回表数据，形如：[(1, "张三", 18), (2, "李四", 20)]
                rows = cursor.fetchall()

                # 把每一行数据从元组转成 CSV 行文本
                # 例如：(1, "张三", 18) -> "1,张三,18"
                results = [",".join(map(str, row)) for row in rows]

                # columns 组成 CSV 头部，rows 组成 CSV 数据体
                # 最终返回：
                # id,name,age
                # 1,张三,18
                header_str = ",".join(columns)
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error:
        return "查询出现异常，请检查数据库连接或表配置。"


@tool
def execute_sql_query(query) -> str:
    """
    执行自定义 SQL 查询

    切记：执行之前，需要通过 list_sql_tables 明确真实表名，
    再通过 get_table_data 明确表结构和数据格式。
    适合多表关联、筛选、聚合、排序等复杂查询。
    :param query: 要执行的自定义 SQL 语句
    :return: CSV 格式数据
             1. 第一行是列信息，列之间使用英文逗号分隔
             2. 第二行开始是表数据，值之间也使用英文逗号分隔
             3. 行和行之间使用 \n 分隔
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n
    """
    # 先在工具层完成校验，拒绝时不读取连接配置，也不会建立 MySQL 连接。
    try:
        _validate_readonly_query(query)
    except ValueError as e:
        return f"拒绝执行：仅允许单条只读 SQL 查询。原因：{str(e)}"

    # 埋点：记录通过只读校验的 SQL，便于教学时观察是否真的落到了正确表字段上
    monitor.report_tool(
        tool_name="数据库表数据查询工具：execute_sql_query", args={"query": query}
    )

    # 获取数据库参数
    config = get_db_config()

    # 自定义查询和 get_table_data 的结果处理逻辑一致：
    # 执行 SQL -> 读取 description 得到列名 -> fetchall 得到数据 -> 拼成 CSV 返回
    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                # 到达此处的 query 已通过单语句只读白名单校验。
                cursor.execute(query)

                # 非查询类 SQL 没有结果集描述，这里统一返回提示，避免工具调用直接抛错给模型
                description = cursor.description
                if not description:
                    return f"执行自定义 SQL 语句没有查询结果，SQL 为：{query}"
                # description => [("列1", ...), ("列2", ...)]
                columns = [desc[0] for desc in description]

                # rows => [(值1, 值2), (值1, 值2)]
                rows = cursor.fetchall()

                # 每行元组统一转为逗号分隔文本，便于模型读取和后续整理
                results = [",".join(map(str, row)) for row in rows]

                # 第一行是列名，后续是查询数据
                header_str = ",".join(columns)
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error as e:
        return f"查询出现异常：{str(e)}"


if __name__ == "__main__":
    # 本地调试入口：直接运行本文件可验证 .env 中的 MySQL 连接配置是否可用
    print(
        execute_sql_query.invoke(
            {
                "query": "SELECT * FROM `drugs` dgs join sales_records srd on dgs.drug_id = srd.drug_id"
            }
        )
    )
