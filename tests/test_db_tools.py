import unittest
from unittest.mock import MagicMock, call, patch

from app.tools import db_tools


class ReadonlySqlGuardTests(unittest.TestCase):
    def test_allows_supported_readonly_statements(self):
        allowed_queries = [
            "SELECT COUNT(*) AS drug_count FROM drugs;",
            "SHOW TABLES;",
            "DESCRIBE drugs;",
            "DESC drugs;",
            "EXPLAIN SELECT * FROM drugs;",
            "WITH x AS (SELECT drug_id FROM drugs) SELECT * FROM x;",
            "SELECT 'a;b' AS text_with_semicolon;",
            "/* read only */ -- continue\n SELECT 1; # trailing comment",
        ]

        for query in allowed_queries:
            with self.subTest(query=query):
                self.assertIsNone(db_tools._validate_readonly_query(query))

    def test_rejects_writes_ddl_multi_statement_and_high_risk_queries(self):
        rejected_queries = [
            "UPDATE drugs SET brand_name='x';",
            "DELETE FROM drugs;",
            "DROP TABLE drugs;",
            "INSERT INTO drugs (drug_id) VALUES (1);",
            "SELECT * FROM drugs; DROP TABLE drugs;",
            "CALL dangerous_proc();",
            "SELECT * FROM drugs INTO OUTFILE '/tmp/x';",
            "SELECT * FROM drugs INTO DUMPFILE '/tmp/x';",
            "/* comment */ UPDATE drugs SET brand_name='x';",
            "UpDaTe drugs SET brand_name='x';",
            "WITH x AS (SELECT drug_id FROM drugs) DELETE FROM drugs;",
            "/*!50000 UPDATE drugs SET brand_name='x' */;",
            "SELECT * FROM drugs FOR UPDATE;",
            "SELECT LOAD_FILE('/etc/passwd');",
        ]

        for query in rejected_queries:
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    db_tools._validate_readonly_query(query)

    def test_rejected_query_never_connects_to_mysql(self):
        rejected_queries = [
            "UPDATE drugs SET brand_name = brand_name WHERE 1 = 0;",
            "DELETE FROM drugs;",
            "DROP TABLE drugs;",
            "SELECT 1; SELECT 2;",
        ]

        with patch("app.tools.db_tools.connect") as connect_mock:
            for query in rejected_queries:
                with self.subTest(query=query):
                    result = db_tools.execute_sql_query.invoke({"query": query})
                    self.assertTrue(result.startswith("拒绝执行："))
            connect_mock.assert_not_called()

    def test_allowed_query_reaches_cursor_execute(self):
        cursor = MagicMock()
        cursor.description = [("value",)]
        cursor.fetchall.return_value = [(1,)]

        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch("app.tools.db_tools.connect") as connect_mock:
            connect_mock.return_value.__enter__.return_value = connection
            result = db_tools.execute_sql_query.invoke({"query": "SELECT 1;"})

        cursor.execute.assert_called_once_with("SELECT 1;")
        self.assertEqual(result, "value\n1")


class TableNameGuardTests(unittest.TestCase):
    def test_allows_expected_simple_table_names(self):
        allowed_names = [
            "drugs",
            "inventory",
            "sales_records",
            "abc_123",
            "_table",
        ]

        for table_name in allowed_names:
            with self.subTest(table_name=table_name):
                self.assertEqual(
                    db_tools._validate_table_name(table_name), table_name
                )

    def test_invalid_table_names_are_rejected_before_connect(self):
        invalid_names = [
            "drugs;",
            "drugs; DROP TABLE drugs",
            "drugs --",
            "drugs/*",
            "drugs`",
            "drugs WHERE 1=1",
            "deepsearch_db.drugs",
            "information_schema.tables",
            "",
            " ",
            None,
            '"drugs"',
            "'drugs'",
            "../drugs",
            "药品",
        ]

        with patch("app.tools.db_tools.connect") as connect_mock:
            for table_name in invalid_names:
                with self.subTest(table_name=table_name):
                    result = db_tools.get_table_data.invoke(
                        {"table_name": table_name}
                    )
                    self.assertEqual(result, "拒绝执行：非法表名。")
            connect_mock.assert_not_called()

    def test_nonexistent_table_stops_after_parameterized_metadata_check(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = None

        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch("app.tools.db_tools.connect") as connect_mock:
            connect_mock.return_value.__enter__.return_value = connection
            result = db_tools.get_table_data.invoke(
                {"table_name": "not_existing_table"}
            )

        self.assertEqual(result, "表不存在或不允许访问。")
        cursor.execute.assert_called_once_with(
            db_tools._TABLE_EXISTS_QUERY,
            ("deepsearch_db", "not_existing_table"),
        )
        cursor.fetchall.assert_not_called()

    def test_existing_tables_use_parameterized_check_and_quoted_select(self):
        for table_name in ["drugs", "inventory", "sales_records"]:
            with self.subTest(table_name=table_name):
                cursor = MagicMock()
                cursor.fetchone.return_value = (1,)
                cursor.description = [("id",)]
                cursor.fetchall.return_value = [(1,)]

                connection = MagicMock()
                connection.cursor.return_value.__enter__.return_value = cursor

                with patch("app.tools.db_tools.connect") as connect_mock:
                    connect_mock.return_value.__enter__.return_value = connection
                    result = db_tools.get_table_data.invoke(
                        {"table_name": table_name}
                    )

                cursor.execute.assert_has_calls(
                    [
                        call(
                            db_tools._TABLE_EXISTS_QUERY,
                            ("deepsearch_db", table_name),
                        ),
                        call(f"SELECT * FROM `{table_name}` LIMIT 100"),
                    ]
                )
                self.assertEqual(cursor.execute.call_count, 2)
                self.assertEqual(result, "id\n1")


if __name__ == "__main__":
    unittest.main()
