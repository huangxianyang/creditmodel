##-*-utf-8-*-
#python3.6
#filename Preprocessing.py
#特征工程
import function.utils as F
import scorecardpy as sc
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor #共线性检验
from sklearn.linear_model import LogisticRegression

class ScoreCard:
    """评分卡"""
    def __init__(self,data,cat_feature,num_feature):
        """
        全局参数
        :param data: dataframe 清洗完成的train Dataset
        :param cat_feature: list 分类特征
        :param num_feature: list 数值特征
        """
        self.data = data
        self.cat_feature = cat_feature
        self.num_feature = num_feature


    def cat_feature_bin(self,label="target",special_values=None,min_perc_fine_bin=0.01,
                        stop_limit=0.1,min_perc_coarse_bin=0.05,max_num_bin=5,method="tree"):
        """
        基于模型的分类变量组合
        :param label:str 标签
        :param special_values: list or dict 不参与组合的特殊值
        :param min_perc_fine_bin: float 子节点最小占比,默认0.01
        :param stop_limit: float 停止分箱基尼值,默认0.1
        :param min_perc_coarse_bin: float 叶子节点占比,默认0.05
        :param max_num_bin: int 最大组合数 默认5
        :param method: str 方法,默认 tree 可选 chimerge 卡方分箱
        :return: dict 分箱结果及IV
        """
        t0 = time.clock()
        var_IV,bin_dict = {},{}
        self.data[self.cat_feature] = self.data[self.cat_feature].astype("str") #防止误将分类变量当做连续变量处理

        for col in self.cat_feature:
            print('we are processing {}'.format(col))
            bin_dict[col] = sc.woebin(self.data, x=col, y=label, special_values=special_values, min_perc_fine_bin=min_perc_fine_bin,
                                  stop_limit=stop_limit, min_perc_coarse_bin=min_perc_coarse_bin,
                                      max_num_bin=max_num_bin, method=method)[col] #最优组合字典
            var_IV[col] = bin_dict[col]["total_iv"].unique()[0] #特征IV保存
        print("处理{}个分类变量组合,耗时:{}".format(len(self.cat_feature),time.clock()-t0))

        return bin_dict,var_IV

    def num_feature_bin(self,label="target",special_values=None,min_perc_fine_bin=0.01,
                        stop_limit=0.1,min_perc_coarse_bin=0.05,max_num_bin=5,method="tree",no_monoto_feature=None):
        """
        基于卡方或决策树连续变量分箱,
        对于连续型变量，处理方式如下：
        1.利用决策树模型或卡方分箱法将变量分成n个组
        2.检查坏样本率的单带性，如果发现单调性不满足，就进行合并，直到满足单调性
        :param label:str 标签
        :param special_values: list or dict 不参与组合的特殊值
        :param min_perc_fine_bin: float 子节点最小占比,默认0.01
        :param stop_limit: float 停止分箱基尼值,默认0.1
        :param min_perc_coarse_bin: float 叶子节点占比,默认0.05
        :param max_num_bin: int 最大组合数 默认5
        :param method: str 方法,默认 tree 可选 chimerge 卡方分箱
        :param no_monoto_feature list or None 不参与单调性检验变量
        :return: dict 分箱结果及IV
        """
        t0 = time.clock()
        var_cutoff = {}
        bin_dict = {}
        var_IV = {}
        for col in self.num_feature:
            try:
                if col not in no_monoto_feature:
                    print("{} is in processing".format(col))
                    # (1),卡方或决策树分箱法进行分箱，并且保存每一个分割的端点。例如端点=[10,20,30]表示将变量分为x<10,10<x<20,20<x<30和x>30.
                    cutOffPoints = sc.woebin(self.data[[col, label]], x=col, y=label, special_values=special_values,
                                             min_perc_fine_bin=min_perc_fine_bin,stop_limit=stop_limit,
                                             min_perc_coarse_bin=min_perc_coarse_bin, max_num_bin=max_num_bin,
                                             method=method)[col]["breaks"].tolist()

                    if 'inf' in cutOffPoints:
                        cutOffPoints.remove("inf")

                    cutOffPoints = [float(i) for i in cutOffPoints] #切分点

                # (2), 单调性检查
                    col1 = col + '_Bin'  # 检验单调性
                    self.data[col1] = self.data[col].map(lambda x: F.AssignBin(x, cutOffPoints=cutOffPoints,
                                                                               special_attribute=special_values))
                    BRM = F.BadRateMonotone(self.data, col1, label, special_attribute=special_values) #是否单调
                    if not BRM:
                        if special_values == []:
                            bin_merged = F.Monotone_Merge(self.data, label, col1)
                            removed_index = []
                            for bin in bin_merged:
                                if len(bin) > 1:
                                    indices = [int(b.replace('Bin ', '')) for b in bin]
                                    removed_index = removed_index + indices[0:-1]
                            removed_point = [cutOffPoints[k] for k in removed_index]
                            for p in removed_point:
                                cutOffPoints.remove(p)
                            var_cutoff[col] = cutOffPoints
                        else:
                            cutOffPoints2 = [i for i in cutOffPoints if i not in special_values]
                            temp = self.data.loc[~self.data[col].isin(special_values)]
                            bin_merged = F.Monotone_Merge(temp, label, col1)
                            removed_index = []
                            for bin in bin_merged:
                                if len(bin) > 1:
                                    indices = [int(b.replace('Bin ', '')) for b in bin]
                                    removed_index = removed_index + indices[0:-1]
                            removed_point = [cutOffPoints2[k] for k in removed_index]
                            for p in removed_point:
                                cutOffPoints2.remove(p)
                            cutOffPoints2 = cutOffPoints2 + special_values
                            var_cutoff[col] = cutOffPoints2 #单调性检验结果

                ####################
                #最终分箱
                ####################
                bin_dict[col] = sc.woebin(self.data, x=col, y=label, breaks_list=var_cutoff, special_values=special_values,
                                          min_perc_fine_bin=min_perc_fine_bin, stop_limit=stop_limit,
                                          min_perc_coarse_bin=min_perc_coarse_bin, max_num_bin=max_num_bin,
                                          method=method)[col]
                # 保存IV
                var_IV[col] = bin_dict[col]["total_iv"].unique()[0]

            except:
                print("变量异常:", col)

        print("总用时{}分钟".format((time.clock() - t0) / 60))

        return bin_dict,var_IV

    def self_bin_dict(self,x,special_values=None,max_num_bin=5,method="tree"):
        """
        手动分箱
        :param x:str 待分箱变量
        :param special_values: list or dict 特殊值
        :param max_num_bin: 最大分箱数
        :param method: 分箱方法
        :return: IV bin_dict
        """
        var_IV = {}
        bin_dict = sc.woebin(dt=self.data,x=x,special_values=special_values,max_num_bin=max_num_bin,method=method)
        var_IV[x] = bin_dict[x]["total_iv"].unique()[0] #IV提取
        return bin_dict,var_IV


    def hight_iv(self,var_iv,threshold=0.02,xlabel=None, figsize=(15,7), is_save=False,path="./"):
        """
        选择IV高于阈值的变量, 一般说来，信息值0.02以下表示与目标变量相关性非常弱。
        0.02-0.1很弱；0.1-0.3一般；0.3-0.5强；0.5-1很强,1以上异常,单独关注
        :param var_iv: dict 特征信息值字典
        :param threshold: float 阈值
        :param path:文件存储地址
        :param model:
        :return:
        """

        high_IV = {k: v for k, v in var_iv.items() if v >= threshold}
        high_IV_df = pd.DataFrame.from_dict((i, j) for i, j in high_IV.items()).sort_values(by=1, ascending=False)
        high_IV_df.set_index(0, inplace=True)
        if is_save: #保存IV图片及IV表
            F.draw_IV(IV_dict=high_IV, path=path, xlabel=xlabel, figsize=figsize, is_save=is_save)
            high_IV_df.to_excel(path+"high_iv_df.xlsx",index=False)
        return high_IV

    def feature_importance(self,n_estimators,target,is_dram=False):
        """
        基于随机森林特征权重
        --------------------------
        parameter:
                 n_estimators: 随机森林树量
                 target: str
                 is_dram:bool 作图
        return:
              feature_importance: dict and importance draw
        """
        data = self.data
        cat_feature  = self.cat_feature
        num_feature = self.num_feature
        if cat_feature:
            for i in cat_feature:
                data[i] = pd.factorize(data[i])[0] #分类变量数值化

        data[num_feature] = data[num_feature].astype("float")

        X = data[cat_feature+num_feature]
        y= data[target]

        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)  # 构建分类随机森林分类器
        rf.fit(X, y)  # 对自变量和因变量进行拟合

        # feature importances dict
        importances = rf.feature_importances_
        feat_names = X.columns
        feature_importance = dict()
        for k, v in zip(feat_names, importances):
            feature_importance[k] = v

        # 可视化
        if is_dram:
            plt.style.use('fivethirtyeight')
            plt.rcParams['figure.figsize'] = (12, 6)
            sns.set_style("darkgrid", {"font.sans-serif": ["simhei", "Arial"]})
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(15, 6))
            plt.title("feature importance")
            plt.bar(range(len(indices)), importances[indices], color='lightblue', align="center")
            plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
            plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical', fontsize=14)
            plt.xlim([-1, len(indices)])
            plt.savefig("feature_importance.png")

        return feature_importance

    def woe_transfer(self,select_feature,bin_dict,label):
        """
        WOE编码
        :param label: str 标签
        :param select_feature: list 选择特征
        :param bin_dict: dict 特征组合字典
        :return: data_woe
        """
        data = self.data
        #woe编码前缺失检查
        print("woe编码前数据集缺失:",data.isnull().values.any())

        data_woe = sc.woebin_ply(data[select_feature + [label]], bin_dict)

        #woe编码后缺失检查
        print("转换后训练集缺失:", data_woe.isnull().values.any())
        if data_woe.isnull().values.any():
            print("woe transfer error")
        else:
            return data_woe

    def collinear_check(self,select_feature,data_woe,cor_threshold,is_draw=False,figsize=(12,12)):

        # 多重共线性
        # 1,将候选变量按照IV进行降序排列
        # 2，计算第i和第i+1的变量的线性相关系数
        # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
        deleted_index = []
        select_feature_sort = sorted(select_feature.items(),key=lambda x:x[1],reverse=True)
        cnt_vars = len(select_feature_sort)
        for i in range(cnt_vars):
            if i in deleted_index:
                continue
            x1 = select_feature_sort[i][0] + "_woe"
            #检查多重共线性
            for j in range(cnt_vars):
                if i == j or j in deleted_index:
                    continue
                y1 = select_feature_sort[j][0] + "_woe"
                roh = np.corrcoef(data_woe[x1], data_woe[y1])[0, 1]
                if abs(roh) > cor_threshold:  # 控制相关系数阈值
                    x1_IV = select_feature_sort[i][1]
                    y1_IV = select_feature_sort[j][1]
                    if x1_IV > y1_IV:
                        deleted_index.append(j)
                    else:
                        deleted_index.append(i)
        # 去除共线性特征
        multi_analysis_vars_1 = [select_feature_sort[i][0] + "_woe" for i in range(cnt_vars) if i not in deleted_index]

        # 多变量分析：VIF
        X = np.matrix(data_woe[multi_analysis_vars_1])

        VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_VIF = max(VIF_list)
        print("最大方差膨胀因子")
        #相关性可视化
        if is_draw:
            F.get_corr(data=data_woe,figsize=figsize)

        return multi_analysis_vars_1

    def caculate_vif(self,data_woe, thres):
        """
        计算方差膨胀因子VIF值
        ---------------------------------------
        parameter:
                 data_woe: pandas dataframe
                 thres : float or int, vif threshold
        return:
               keep_feature : dataframe
               remove_feature : dataframe
               keep_feature_list : list

        """
        vif = pd.DataFrame()
        vif["feature"] = list(data_woe.columns)
        vif["vif_value"] = [variance_inflation_factor(data_woe.values, i) for i in range(data_woe.shape[1])]
        keep_feature = vif.loc[vif["vif_value"] < thres, :]
        remove_feature = vif.loc[vif["vif_value"] >= thres, :]
        keep_feature_list = keep_feature["feature"].tolist()
        return keep_feature, remove_feature, keep_feature_list

    def select_feature_L1(self,data_woe,in_model_var,target,C=1):
        """
        基于L1正则选择特征
        :param data_woe:dataframe woe编码后的数据集合
        :param in_model_var: list 模型特征
        :param target: str 标签
        :param C: float 正则力度 >0
        :return:var_coe_dict 变量系数
        """
        LR = LogisticRegression(C=C, penalty='l1', class_weight="balanced")
        LR.fit(data_woe[in_model_var], data_woe[target])

        # 模型系数
        paramsEst = pd.Series(LR.coef_.tolist()[0], index=in_model_var)
        var_coe_dict = paramsEst.to_dict()
        # 模型变量选择
        return var_coe_dict

    def remove_no_explan(self,all_feature,remove_feature):
        """
        移除不可解释变量
        :param all_feature:list
        :param remove_feature: list
        :return: last_feature
        """
        last_feature = [i for i in all_feature if i not in remove_feature]
        return last_feature
