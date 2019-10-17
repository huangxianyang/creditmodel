#-*-utf-8-*-
#python3.6
#filename Preprocessing.py
#data preprocessing function
import pandas as pd
import numpy as np
import random as rd
import numbers

class Data_sample:
    """
    数据抽样
    """
    def __init__(self,data):
        """
        全局变量
        """
        self.data = data

    def balance_sample(self, label):
        """
        平衡类别抽样
        :param label: 分层变量
        :return: dataframe 抽样结果
        """
        diff_case = pd.DataFrame(self.data[label]).drop_duplicates([label])
        result = []
        result = pd.DataFrame(result)
        for i in range(len(diff_case)):
            k = np.array(diff_case)[i]
            data_set = self.data[self.data[label] == k[0]]
            nrow_nb = data_set.iloc[:, 0].count()
            data_set.index = range(nrow_nb)
            index_id = rd.sample(range(nrow_nb), int(nrow_nb * 1))
            result = pd.concat([result, data_set.iloc[index_id, :]], axis=0)
        new_data = pd.Series(result['label']).value_counts()
        new_data = pd.DataFrame(new_data)
        new_data.columns = ['cnt']
        k1 = pd.DataFrame(new_data.index)
        k2 = new_data['cnt']
        new_data = pd.concat([k1, k2], axis=1)
        new_data.columns = ['id', 'cnt']
        max_cnt = max(new_data['cnt'])
        k3 = new_data[new_data['cnt'] == max_cnt]['id']
        result = result[result[label] == k3[0]]
        return result

    def random_sample(self,n=None,frac=0.5):
        """
        随机抽样
        :param n:抽样样本量
        :param frac: float 抽样样本占比
        :return:dataframe 抽样结果
        """
        if n != None:
            new_data = self.data.sample(n = n)
        elif frac != None:
            new_data = self.data.sample(frac=frac)
        else:
            pass
        return new_data

class DataConcat:
    """
    数据连接
    """
    def __init__(self):
        """
        全局变量
        """
    def concat(self,data1,data2,axis=0):
        """
        数据叠加
        :param data1: dataframe
        :param data2: dataframe
        :param axis:0 or 1 叠加方式
        :return: data
        """
        data = pd.concat([data1,data2],axis=axis)
        return data

    def merge(self,data1,data2,how="inner",on=None,left_on=None,right_on=None):
        """
        表连接
        :param data1:dataframe
        :param data2: dataframe
        :param how: 连接方式 inner left right outer default inner
        :param on: 连接键,if on is None, left_no and right_on must be not None
        :param left_on: 左连接键值
        :param right_on:右连接键值
        :return:dataframe连接结果
        """
        data = pd.merge(data1,data2,how=how,on=on,left_on=left_on,right_on=right_on)
        return data

class PreprocessData:
    """
    数据预处理
    """
    def __init__(self,data):
        """
        全局变量
        """
        self.data = data
    def split_col(self,no_split,min_size=5):
        """
        变量分类
        :param no_split: list 不参与分类变量
        :param min_size: int 连续变量最小类别数,默认5
        :return: num_col and cat_col list 分类结果
        """
        data = self.data
        num_col,cat_col = [],[]
        for i in [i for i in data.columns.tolist() if i not in no_split]:
            unique_list = data[i].unique().tolist()
            if len(unique_list) > min_size and isinstance(unique_list[0], numbers.Real):
                num_col.append(i)
            else:
                cat_col.append(i)
        return num_col,cat_col

    def Re_str(self, num_col):
        """
        识别连续变量列中的特殊字符串
        :param num_col:list 需要检测的变量
        :return:set 特殊值序列
        """
        data = self.data
        special_str = set()
        for var in num_col:
            for i in list(data[var].unique()):
                try:
                    if str(i) == "nan" or int(i):
                        pass
                except:
                    special_str.add(i)
        special_str = [i for i in special_str]
        return special_str

    def replace_data(self,old,new,replace_dict=None):
        """
        数据字段替换
        :param old: object,待替换字符 str int float list
        :param new:object 替换序列 str int float list
        :param replace_dict:dict 替换字典
        :return: 替换结果
        """
        try:
            data = self.data
            if replace_dict != None:
                data = data.replace(replace_dict)
            else:
                data = data.replace(old,new)
            return data
        except:
            print("old and new 参数不匹配")

    def dropDuplicate(self,axis=0):
        """
        删除重复行或重复列
        :param axis: 0 or 1 0表示删除重复行,1表示删除重复列
        :return: 删除结果
        """
        data = self.data
        if axis ==0:
            #删除重复行
            data = data.drop_duplicates()
        else: #删除重复列
            data = data.T.drop_duplicates().T
        return data

    def count_miss_mode(self,col_list):
        """
        缺失率和众数率统计
        :param col_list:统计字段
        :return: dict 缺失率和众数率字典
        """
        data = self.data
        miss_mode = {}
        miss_mode["miss_rate"],miss_mode["mode_rate"] = {},{}

        for i in col_list:
            miss_rate = data[i].isnull().sum() / len(data)  # 缺失率
            miss_mode["miss_rate"][i] = miss_rate
            try:
                mode_values = data[i].mode()[0]  # 众数
            except:
                print("变量{}不存在众数".format(i))
                mode_values = None
            mode_rate = len(data.loc[data[i] == mode_values, :]) / len(data)  # 众数率
            miss_mode["mode_rate"][i] = mode_rate
        return miss_mode

    def drop_nan_mode(self,nan_percent,mode_percent,col_list,drop=False):
        """
        删除缺失率和众数率较大的变量
        :param nan_percent: float 0~1 缺失率
        :param mode_percent: float 0~1 众数率
        :param col_list: list 作用变量列表
        :param drop:bool 是否删除
        :return: dataframe
        """
        data = self.data
        del_col = []
        miss_model_rate = self.count_miss_mode(col_list)
        for i in col_list:
            miss_rate = miss_model_rate["miss_rate"][i] #缺失率
            mode_rate = miss_model_rate["mode_rate"][i] #众数率
            if miss_rate >= nan_percent:
                del_col.append(i)
            elif mode_rate >= mode_percent:
                del_col.append(i)

        #是否删除操作
        if drop:
            for i in del_col:
                try:
                    del data[i]
                except:
                    print("变量:{}已删除".format(i))

            return data,del_col
        else:
            return del_col

    def fill_nan(self,value=None,method=None,col_list=None,min_miss_rate=0.05):
        """
        缺失值填充
        :param value:dict str int float 列填充值字典 默认None
        :param method:str,方法 backfill, bfill, pad, ffill, None, mode,mean,special_value,采用均值填充的字段必须为连续变量
        :param col_list:填充列,特殊方式填充
        :param min_miss_rate,最小缺失率
        :return: dataframe
        """
        data = self.data
        if method in ["backfill", "bfill", "pad", "ffill"]:
            data = data.fillnan(method=method)
        elif method in[None,"special_value"]: #特殊值填充或列字典填充
            data = data.fillnan(value=value)
        elif method in ["mode","mean"] and len(col_list)>0: #特殊填充方法
            miss_model_rate = self.count_miss_mode(col_list)
            ###特殊填充方法
            for i in col_list:

                try:
                    mode = data[i].mode()[0]
                except:
                    mode = -9999

                if value != None:
                    if miss_model_rate["miss_rate"][i] > min_miss_rate:
                        data[i] = data[i].fillna(value=value)
                    elif method =="mode":
                        data[i] = data[i].fillna(value=mode)
                    elif method =="mean":
                        data[i] = data[i].fillna(value=data[i].mean())
                elif value == None:
                    if method == "mode":
                        data[i] = data[i].fillna(value=mode)
                    elif method =="mean":
                        data[i] = data[i].fillna(value=data[i].mean())
        else:
            data = data.fillnan(-9999)
        return  data

    def factor_data(self,col_list=None):
        """
        字符变量数值化
        :param col_list:需要数值化字段
        :return:data, factor_dict 数值化映射
        """
        data = self.data
        factorize_dict = {}
        if col_list == None:
            for var in data.select_dtypes(include=["object", "bool"]).columns.tolist():
                factorize_dict[var] = {}
                for i in np.arange(len(data[var].unique())):
                    factorize_dict[var][pd.factorize(data[var])[1][i]] = i #数值化映射字典
                data[var] = pd.factorize(data[var])[0] #数值化
        else:
            for var in col_list:
                factorize_dict[var] = {}
                for i in np.arange(len(data[var].unique())):
                    factorize_dict[var][pd.factorize(data[var])[1][i]] = i  # 数值化映射字典
                data[var] = pd.factorize(data[var])[0]  # 数值化
        return data,factorize_dict #数值数据, 映射字典

