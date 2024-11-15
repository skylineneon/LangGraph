import sqlite3

class SQLiteDB:
    
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        
    def execute(self, sql, params=None):
        """
        执行单条SQL语句
        :param sql: SQL语句
        :param params: SQL语句中的参数
        """
        self.cursor = self.conn.cursor()
        if params:
            self.cursor.execute(sql, params)
        else:
            self.cursor.execute(sql)
        self.conn.commit()
        


    def fetchall(self,sql):
        """
        获取所有查询结果
        """
        self.cursor = self.conn.cursor()
        self.cursor.execute(sql)
        return self.cursor.fetchall()


    def close(self):
        """
        关闭数据库连接
        """
        self.conn.close()

sqlite_client = SQLiteDB("/root/workspace/LangGraph/数据库代理/company.db")
# print(sqlite_client.fetchall("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5"))

# # 使用示例
# if __name__ == "__main__":
#     db = SQLiteDB("/root/workspace/LangGraph/数据库代理/company.db")

#     db.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
#     print(db.fetchall())

#     # 关闭数据库连接
#     db.close()