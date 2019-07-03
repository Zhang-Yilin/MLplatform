import pymssql
import pandas as pd

class SQLServer:
    def __init__(self, host, port, user, pwd, db):
        #self.conn = pymssql.connect(host=host, port=port, user=user, password=pwd, database=db, charset="utf8")
        self.host=host
        self.port=port
        self.user=user
        self.pwd=pwd
        self.db=db

    def df_read_sqlserver(self, table, cols=None):
        '''
        取数
        :rtype: object
        :param table: 数据库表名
        :return: 所选数据
        '''
        conn = pymssql.connect(host=self.host, port=self.port, user=self.user, password=self.pwd, database=self.db, charset="utf8")
        df = None
        if cols is None:
            sql = "SELECT * FROM {table}".format(table=table)
        else:
            sql = "SELECT {col} FROM {table}".format(col=','.join(cols), table=table)
       # print(sql)
        try:
            df = pd.read_sql(sql, conn)
           #print(df)
        except Exception as e:
            print(e)
        finally:
            conn.close()
        return df

    def df_write_sqlserver(self, table, df, cols):
        '''
        存数
        :param table: 数据库表名
        :param df: 要存储数据，dataframe格式
        :return: 存储标识
        '''
        conn = pymssql.connect(host=self.host, port=self.port, user=self.user, password=self.pwd, database=self.db,charset="utf8")
        dd = 10
        delsql = "delete from {table}".format(table=table)
        conn.cursor().execute(delsql)
        conn.commit()
        #print(df.columns.size)
        #inssql = "insert into {table} ({col},class) values ({s})".format(table=table,col=','.join(cols),s=','.join(["%s"]*(len(cols)+1)))
        inssql = "insert into {table} ({col},class) values ({s})".format(table=table,col=','.join(cols),s=','.join(["%s"]*(df.columns.size)))
        #print(inssql)
        try:
            for i in range(0, len(df) + dd, dd):
                new_df = df.iloc[i:i + dd, :]
                lst = [tuple([str(c) for c in list(dict(row).values())]) for _, row in new_df.iterrows()]
                conn.cursor().executemany(inssql,lst)
                conn.commit()
        except Exception as e:
            print(e)
            return "failed,{e}".format(e=e)
        finally:
            conn.close()
        return "success"


if __name__ == "__main__":
    s = SQLServer(host = "10.24.12.94",port = 1433,user = "sa",pwd = "sa1234!", db = "algonkettle")

    r = s.df_read_sqlserver(table = "dbo.iris",cols = ["sepal_lenth", "sepal_width", "petal_lenth", "petal_width"])
    print(r)