"""文档资产目录的连接约定；各仓库保留自己的表和事务边界。"""

import sqlite3


def connect(database):
    return sqlite3.connect(database, timeout=30)


def initialize(connection):
    connection.execute("PRAGMA journal_mode=WAL")
