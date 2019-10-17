#-*-utf-8-*-
#python3.6
#filename Preprocessing.py
#统计分析指标
import missingno as msno
import pandas_profiling as pandas_pf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab


class BasicStatistics(object):
    """数据概述"""
    def __init__(self,data):
        """
        基本指标统计类
        :param data:dataframe 数据集
        """
        self.data = data
        self.shape = data.shape

    def feature_describe(self,x=None,percentiles=None,include=None):
        """
        字段基本信息
        :param x: str or list 统计变量, 默认None
        :param percentiles: list-like of float 默认 None,[0.25,0.5,0.75]
        :param include: 'all', list-like of dtypes or None (default)
        :return:feature_describe 字段基本信息
        """
        data = self.data
        if x == None:
            feature_describe = data.describe(percentiles=percentiles,include=include).T
        else:
            feature_describe = data[x].describe(percentiles=percentiles,include=include).T
        return feature_describe

    def feature_dtypes(self):
        """
        特征属性
        :return:字段类型字典
        """
        data = self.data
        dtypes = data.dtypes.to_dict()
        return dtypes



    def data_report(self,path,filename):
        """
        数据报告
        :param path: str 保存文件路径
        :param filename: str 保存文件名
        :return:数据报告 html格式
        """
        data =self.data
        profile = pandas_pf.ProfileReport(data)
        profile.to_file(outputfile=path+filename+".html")

class Plot(object):
    """
    作图
    """

    def draw_pie(self,s,filename):
        """
        字符型变量饼图
        -------------------------------------
        Params
        s: pandas Series
        lalels:labels of each unique value in s
        dropna:bool obj
        filename: 保存图路径及文件名
        -------------------------------------
        Return
        show the plt object
        """
        counts = s.value_counts(dropna=True)
        labels = counts.index
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(counts, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
        ax.axis('equal')
        ax.set_title(r'pie of {}'.format(s.name))
        plt.savefig(filename)
        plt.show()

    def draw_bar(self,s, filename, x_ticks=None, pct=False, horizontal=False):
        """
        字符型变量条形图
        -------------------------------------------
        Params
        s: pandas Series
        x_ticks: list, ticks in X axis
        pct: bool, True means trans data to odds
        dropna: bool obj,True means drop nan
        horizontal: bool, True means draw horizontal plot
        -------------------------------------------
        Return
        show the plt object
        """
        counts = s.value_counts(dropna=True)
        if pct == True:
            counts = counts / s.shape[0]
        ind = np.arange(counts.shape[0])
        plt.figure(figsize=(8, 6))
        if x_ticks is None:
            x_ticks = counts.index

        if horizontal == False:
            p = plt.bar(ind, counts)
            plt.ylabel('frequecy')
            plt.xticks(ind, tuple(counts.index))
        else:
            p = plt.barh(ind, counts)
            plt.xlabel('frequecy')
            plt.yticks(ind, tuple(counts.index))
        plt.title('Bar plot for %s' % s.name)
        plt.savefig(filename)
        plt.show()

    def drawHistogram(self,s, num_bins,filename):
        """
        连续变量分布图
        ---------------------------------------------
        Params
        s: pandas series
        num_bins: number of bins
        save: bool, is save?
        filename png name
        ---------------------------------------------
        Return
        show the plt object
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        mu = s.mean()
        sigma = s.std()

        n, bins, patches = ax.hist(s, num_bins, normed=1, rwidth=0.95, facecolor="blue")

        y = mlab.normpdf(bins, mu, sigma)
        ax.plot(bins, y, 'r--')
        ax.set_xlabel(s.name)
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram of %s: $\mu=%.2f$, $\sigma=%.2f$' % (s.name, mu, sigma))
        plt.savefig(filename)
        plt.show()

    def missing(self,data,path,asc=0,figsize=(10,6)):
        """
        缺失可视化
        :param data:dataframe
        :param path:str 保存结果路径及名称
        :param asc: int 统计方法,Matrix(asc=0),BarChart(asc=1),Heatmap(asc=2)
        :param figsize tupe 图片大小
        :return:保存结果
        """
        if asc==0:
           msno.matrix(df=data)
           plt.savefig(path+"miss_nan.png")
        elif asc ==1:
            msno.bar(df=data,figsize=figsize)
            plt.savefig(path + "miss_nan.png")
        else:
            msno.heatmap(df=data,figsize=figsize)

    def plot_scatter(self,data,x1,x2,x1label=None,x2label=None,path="./"):
        """
        散点图
        :param data:dataframe
        :param x1: str x1变量
        :param x2: str x2 变量
        :param x1label: x1 str 标签
        :param x2label: x2 str 标签
        :param path: 保存图路径
        :return:保存图
        """
        plt.scatter(data=data,x=x1,y=x2)
        plt.xlabel(x1label)
        plt.ylabel(x2label)
        plt.savefig(path+"{}-{}.png".format(x1,x2))

    def mult_boxplots(self,data, variable, category,xlabel=None, ylabel=None, title=None):
        """
        箱线图,探索数据分布
        :param df: dataframe
        :param variable: str 统计变量
        :param category: str 分组值
        :param xlabel: str
        :param ylabel: str
        :param title: str
        :return: 保存图
        """
        data[[variable, category]].boxplot(by=category) #作图

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        plt.savefig("{}-{}_box.png".format(variable,category)) #保存