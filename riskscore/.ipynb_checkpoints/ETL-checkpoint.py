#-*-utf-8-*-
#python3.6
#filename ETL.py
#ETL function

from sqlalchemy import create_engine
import pandas as pd
from function.utils import load_pkl
#数据加载及保存类
class LoadData:
    """
    数据源连接
    """
    def __init__(self):
        """
        全局参数
        """
    def load_mysql(self,host,user,password,database,sql):
        """
        从mysql数据库中加载数据
        :param host:str,主机ip,例如'10.0.16.108'
        :param user: str,数据库用户名
        :param password:str,用户密码
        :param database:str,数据库名
        :param sql:str 查询语句
        :return: df,dataframe
        example:
        sql = "select * from view_anti_fraud_data"
        df = loa_data(host="10.0.16.108",user="root",password="root",database="anti_fraud",sql = sql)
        """
        db_info = {'user': user,
           'password': password,
           'host':host,
           'database':database
           }
        engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s/%(database)s?charset=utf8'%\
                               db_info, encoding='utf-8')
        df = pd.read_sql_query(sql, con=engine)
        return df

    def load_local(self,Path="./data/",data ="data.csv",sort="csv",encoding="utf-8",sep=r",",\
                   skiprows = 0, header="infer",names=None,quoting=0,chunksize=None):
        """
        加载本地数据
        :param Path:str 文件路径, 默认./data/
        :param data: str 被加载文件名,默认data.csv
        :param sort: str 被加载文件类型,sas,excel,csv,txt,pkl
        :param sep: str , /t 分隔符
        :param header: str or None 表头,默认第一行
        :param names: list or None 表头名,默认None,当header=None时,names比如为list
        :param skiprows: int 跳过n行,默认0
        :param encoding: str 被加载文件编码,默认 utf-8 选项 gbk utf-8,ANSI
        :param quoting:int 默认0
        :param chunksize: int. batch_data 文件块大小,适用于读大文件,默认为0
        :return:data
        """
        if sort in ["csv","txt"]:
            data = pd.read_csv(filepath_or_buffer=Path+data,sep=sep,header=header,names=names,\
                               skiprows=skiprows,quoting=quoting,encoding=encoding,chunksize=chunksize,low_memory=False)
        elif sort =="excel":
            data = pd.read_excel(io=Path+data,encoding=encoding,chunksize=chunksize)
        elif sort =="sas":
            data = pd.read_sas(filepath_or_buffer=Path+data,encoding=encoding,chunksize=chunksize)
        elif sort =="pkl":
            data = load_pkl(filename=Path+data,model="rb")
        else:
            print("暂时不支持该类文件")
        return data

    def save_data(self,filename,data,sort,save_index=False):
        """
        保存文件
        :param filename:str 保存路径
        :param data: object 保存对象
        :param sort: 保存类型,excel csv pkl
        :param save_index: bool 是否保存索引
        :return:保存成功
        """
        try:
            if sort =="csv":
                data.to_csv(path_or_buf = filename,index=save_index)
            elif sort =="excel":
                data.to_excel(excel_writer = filename,index=save_index)
            elif sort == "pkl":
                load_pkl(filename=filename,model="wb",data=data)
            return "保存成功"
        except:
            print("保存失败")

