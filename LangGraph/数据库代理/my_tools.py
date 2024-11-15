from langchain_core.tools import tool
from dao import sqlite_client
from typing import Annotated
import sqlite3

@tool
def company_query(sql: Annotated[str, "SQL"]):
    """通过SQL语句查询公司数据库的内容"""
    
    _conn = sqlite3.connect("/root/workspace/LangGraph/数据库代理/company.db")
    _cursor = _conn.cursor()
    _cursor.execute(sql)
    _rt = _cursor.fetchall()
    _conn.close()
    return _rt

@tool
def company_execute(sql: Annotated[str, "执行的SQL语句"]):
    """通过SQL语句执行公司数据库的内容的添加，删除，更新"""
    try:
        _conn = sqlite3.connect("/root/workspace/LangGraph/数据库代理/company.db")
        _cursor = _conn.cursor()
        _cursor.execute(sql)
        _conn.commit()
    finally:
        _conn.close()
    return "OK"