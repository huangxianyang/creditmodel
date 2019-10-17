# -*- coding: utf-8 -*-
# 风险策略,稳定性评估PSI,评分决策表,KS
# 单变量PSI

import pandas as pd
import numpy as np
from .utils import InputCondition as IC

def caculate_ks(df, score:str, target:str):
    """
    指标KS,不分组情况下
    :param data:dataframe
    :param score:str 分数或概率
    :param target:str 好坏定义,0,1
    :return:KS
    """
    df = IC.check_df(df)
    score = IC.check_str(score)
    target = IC.check_str(target)

    df = df.sort_values(by=score)
    total_good = len(df.loc[df[target] == 0, :])
    total_bad = len(df.loc[df[target] == 1, :])
    df_good, df_bad = pd.DataFrame(), pd.DataFrame()

    df_good["score"] = df.loc[df[target] == 0, score]
    df_good["sum_good"] = [i + 1 for i in range(len(df.loc[df[target] == 0, target]))]
    df_good["sum_good_rate"] = df_good["sum_good"] / total_good
    df_good = df_good.drop_duplicates(["score"], keep="last")

    df_bad["sum_bad"] = df.loc[df[target] == 1, target].cumsum()
    df_bad["score"] = df.loc[df[target] == 1, score]
    df_bad["sum_bad_rate"] = df_bad["sum_bad"] / total_bad
    df_bad = df_bad.drop_duplicates(["score"], keep="last")

    df = pd.merge(df_bad, df_good, how="outer", on="score")
    df = df.sort_values(by="score")
    df = df.fillna(method="ffill")
    df = df.fillna(0)

    df["KS"] = df["sum_bad_rate"] - df["sum_good_rate"]
    KS = abs(df["KS"]).max()
    return KS, df

def var_psi(select_feature:list,trainData,testData):
    """
    入模变量PSI
    :param select_feature:list 入模变量
    :param trainData:dataframe 训练集
    :param testData: dataframe 测试集
    :return: 变量psi
    """
    select_feature = IC.check_list(select_feature)
    trainData = IC.check_df(trainData)
    testData = IC.check_df(testData)
    psi_dict = {}
    for var in select_feature:
        psi_dict[var] = ((trainData[var].value_counts()/len(trainData))-(testData[var].value_counts()/len(testData)))*\
                        (np.log((trainData[var].value_counts()/len(trainData))/(testData[var].value_counts()/len(testData))))
    psi_dict_var = {}
    # 输出每个变量的PSI
    for var in select_feature:
        psi_dict_var[var] = psi_dict[var].sum()
    var_psi_df = pd.DataFrame(psi_dict_var,index=[0]).T
    return var_psi_df


def bin_list(low_line:int,up_line:int,step:int):
    """
    分组
    :param low_line:int 下限
    :param up_line: int 上限
    :param step: int 步长
    :return: list 分组列表
    """
    bin = [] # 分组方法
    bin.append(float("-inf")) # 极小值
    for i in range(int(round((up_line-low_line)/step,0))):
        bin.append(low_line+step*i)
    bin.append(up_line)
    bin.append(float("inf")) # 极大值

    return bin


def stragety_score(score_df,low_line:int,up_line:int,step:int,score:str="score",label:str="target"):
    """
    分组函数
    :param score_df:dataframe 包括score 和target
    :param score: str 分数字段 默认 score
    :param label : str 目标字段 默认 target
    :param low_line:int 分组下限
    :param up_line: int 分组上限
    :param step int 步长
    :return: 分组结果
    """
    score_df = IC.check_df(score_df)
    #分组
    bin = bin_list(low_line,up_line,step) #分组列表
    # total 每个分数段总计
    temp_df = score_df[score]
    total_cut = pd.cut(x=temp_df,bins=bin,right=False)
    total_df = pd.value_counts(total_cut) # 每个分数段总计数
    total_sum = len(total_cut) # 总计

    # good 每个分数段的好客户
    temp_df = score_df.loc[score_df[label]==0,score]
    good_cut = pd.cut(x=temp_df,bins=bin,right=False)
    good_df = pd.value_counts(good_cut) # 每个分数段好客户计数
    good_sum = len(good_cut) # 好客户总数

    # bad 每个分数段的坏客户
    temp_df = score_df.loc[score_df[label]==1,score]
    bad_cut = pd.cut(x=temp_df,bins=bin,right=False)
    bad_df = pd.value_counts(bad_cut) # 每个分数段的坏客户计数
    bad_sum = len(bad_cut) # 坏客户总数

    # 总计,好,坏 计数合并
    df = pd.concat([total_df,good_df,bad_df],axis=1)
    df.columns = ["total","good","bad"]
    # 累积
    df_cumsum = df.cumsum()
    df_cumsum.columns = ["sum_total","sum_good","sum_bad"] # 累积总,累积好,累积坏 计数
    df = pd.concat([df,df_cumsum],axis=1)

    # 累积率
    df["sum_total_rate"] = df["sum_total"]/total_sum # 累积坏比率
    df["sum_good_rate"] = df["sum_good"]/good_sum # 累积好比率
    df["sum_bad_rate"] = df["sum_bad"]/bad_sum # 累积坏比率
    df["KS"] = df["sum_bad_rate"] - df["sum_good_rate"] # 区分度ks
    df["bad_rate"] = df["bad"]/df["total"] #每个分数段的坏比率
    df["good_rate"] = df["good"]/df["total"] #每个分数段的好比率

    return df


def score_psi(trainScore,testScore,low_line:int,up_line:int,step:int,score="score"):

    """
    分数PSI
    :param trainScore:df 训练集分数
    :param testScore:df 测试集分数
    :param low_line:int 下限
    :param up_line: int 上限
    :param step: int 步长
    :param score: str 分数字段,默认score
    :return: 返回PSI计算表及PSI指标
    """
    trainScore = IC.check_df(trainScore)
    testScore = IC.check_df(testScore)

    #分组
    bin = bin_list(low_line=low_line,up_line=up_line,step=step)
    train_len = len(trainScore) # 训练集总数
    test_len = len(testScore) # 测试集总数

    train_cut = pd.cut(x=trainScore[score],bins=bin,right=False)  # 训练集分组
    test_cut = pd.cut(x=testScore[score],bins=bin,right=False)  # 测试集分组结果
    train_df = pd.value_counts(train_cut)  # 训练集分组计数
    test_df = pd.value_counts(test_cut)  # 测试集分组计数
    df = pd.concat([train_df,test_df],axis=1) # 合并
    df.columns = ["train","test"]
    df["train_percent"] = df["train"]/train_len #每个分数段的计数比例
    df["test_percent"] = df["test"]/test_len
    df["PSI"] = (df["train_percent"]-df["test_percent"])*np.log(df["train_percent"]/df["test_percent"]) #每个分段的psi
    PSI = df["PSI"].sum()
    return PSI,df
