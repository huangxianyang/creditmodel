#常用函数
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

def load_pkl(filename, model, data=None):
    """
    数据加载与保存
    :param filename:加载或保存对象,格式为.pkl
    :param model: rb or wb 加载或保存模式
    :param data : data.pkl 保存对象
    :return: 返回结果
    """
    if model == "rb+" or model == "rb":
        with open(filename, model) as f:
            data = pickle.load(f)
            f.close()
            return data

    elif model == "wb" or model == "wb+":
        with open(filename, model) as f:
            f.write(pickle.dumps(data, protocol=4))
            f.close()
    else:
        print("not exits filename or input error model")

def groupby_key(data,user_id,content,label):
    """
    根据user_id 合并content 和label列
    :param data: dataframe
    :param user_id:
    :param content: 文本列
    :param label:目标文件
    :return: dataframe
    """
    data[content] = data[content].astype("str")
    content_Series = data.groupby(by=user_id)[content].sum()
    content_df = pd.DataFrame({"user_id":content_Series.index,"content":content_Series.values})
    label_df = data[[user_id,label]].drop_duplicates()
    df= pd.merge(content_df,label_df,on=user_id,how="inner")
    return df


def feature_subgroup(data, index, columns):

    """
    特征分组
    :param index:list
    :param columns:list
    :return: dataframe
    """
    g = data.groupby(index).agg({col: 'nunique' for col in columns})
    if g[g > 1].dropna().shape[0] != 0:
        print("index非唯一值.")
    return data.groupby(index).agg({col: 'max' for col in columns})




######################
##scorecard 常用函数
######################

def AssignBin(x, cutOffPoints, special_attribute=[]):

    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    例如, cutOffPoints = [10,20,30], 对于 x = 7, 返回 Bin 0；对于x=23，返回Bin 2； 对于x = 35, return Bin 3。
    对于特殊值，返回的序列数前加"-"
    '''
    cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
    numBin = len(cutOffPoints2)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints2[0]:
        return 'Bin 0'
    elif x > cutOffPoints2[-1]:
        return 'Bin {}'.format(numBin)
    else:
        for i in range(0,numBin):
            if cutOffPoints2[i] < x <=  cutOffPoints2[i+1]:
                return 'Bin {}'.format(i+1)

def FeatureMonotone(x):
    '''
    特征单调性检验
    Param x: list cut off list
    :return: 返回序列x中有几个元素不满足单调性，以及这些元素的位置。
    例如，x=[1,3,2,5], 元素3比前后两个元素都大，不满足单调性；元素2比前后两个元素都小，也不满足单调性。
    故返回的不满足单调性的元素个数为2，位置为1和2.
    '''
    monotone = [x[i]<x[i+1] and x[i] < x[i-1] or x[i]>x[i+1] and x[i] > x[i-1] for i in range(1,len(x)-1)]
    index_of_nonmonotone = [i+1 for i in range(len(monotone)) if monotone[i]]
    return {'count_of_nonmonotone':monotone.count(True), 'index_of_nonmonotone':index_of_nonmonotone}

def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    标签类别统计函数
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(drop=False, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad / x.total, axis=1)
    dicts = dict(zip(regroup[col],regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


def BadRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    判断单调性函数
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone = FeatureMonotone(badRate)['count_of_nonmonotone']
    if badRateNotMonotone > 0:
        return False
    else:
        return True


def Monotone_Merge(df, target, col):
    '''
    合并方案
    :return:将数据集df中，不满足坏样本率单调性的变量col进行合并，使得合并后的新的变量中，坏样本率单调，输出合并方案。
    例如，col=[Bin 0, Bin 1, Bin 2, Bin 3, Bin 4]是不满足坏样本率单调性的。合并后的col是：
    [Bin 0&Bin 1, Bin 2, Bin 3, Bin 4].
    合并只能在相邻的箱中进行。
    迭代地寻找最优合并方案。每一步迭代时，都尝试将所有非单调的箱进行合并，每一次尝试的合并都是跟前后箱进行合并再做比较
    '''
    def MergeMatrix(m, i,j,k):
        '''
        :param m: 需要合并行的矩阵
        :param i,j: 合并第i和j行
        :param k: 删除第k行
        :return: 合并后的矩阵
        '''
        m[i, :] = m[i, :] + m[j, :]
        m = np.delete(m, k, axis=0)
        return m

    def Merge_adjacent_Rows(i, bad_by_bin_current, bins_list_current, not_monotone_count_current):
        '''
        :param i: 需要将第i行与前、后的行分别进行合并，比较哪种合并方案最佳。判断准则是，合并后非单调性程度减轻，且更加均匀
        :param bad_by_bin_current:合并前的分箱矩阵，包括每一箱的样本个数、坏样本个数和坏样本率
        :param bins_list_current: 合并前的分箱方案
        :param not_monotone_count_current:合并前的非单调性元素个数
        :return:分箱后的分箱矩阵、分箱方案、非单调性元素个数和衡量均匀性的指标balance
        '''
        i_prev = i - 1
        i_next = i + 1
        bins_list = bins_list_current.copy()
        bad_by_bin = bad_by_bin_current.copy()
        not_monotone_count = not_monotone_count_current
        #合并方案a：将第i箱与前一箱进行合并
        bad_by_bin2a = MergeMatrix(bad_by_bin.copy(), i_prev, i, i)
        bad_by_bin2a[i_prev, -1] = bad_by_bin2a[i_prev, -2] / bad_by_bin2a[i_prev, -3]
        not_monotone_count2a = FeatureMonotone(bad_by_bin2a[:, -1])['count_of_nonmonotone']
        # 合并方案b：将第i行与后一行进行合并
        bad_by_bin2b = MergeMatrix(bad_by_bin.copy(), i, i_next, i_next)
        bad_by_bin2b[i, -1] = bad_by_bin2b[i, -2] / bad_by_bin2b[i, -3]
        not_monotone_count2b = FeatureMonotone(bad_by_bin2b[:, -1])['count_of_nonmonotone']
        balance = ((bad_by_bin[:, 1] / N).T * (bad_by_bin[:, 1] / N))[0, 0]
        balance_a = ((bad_by_bin2a[:, 1] / N).T * (bad_by_bin2a[:, 1] / N))[0, 0]
        balance_b = ((bad_by_bin2b[:, 1] / N).T * (bad_by_bin2b[:, 1] / N))[0, 0]
        #满足下述2种情况时返回方案a：（1）方案a能减轻非单调性而方案b不能；（2）方案a和b都能减轻非单调性，但是方案a的样本均匀性优于方案b
        if not_monotone_count2a < not_monotone_count_current and not_monotone_count2b >= not_monotone_count_current or \
                                        not_monotone_count2a < not_monotone_count_current and not_monotone_count2b < not_monotone_count_current and balance_a < balance_b:
            bins_list[i_prev] = bins_list[i_prev] + bins_list[i]
            bins_list.remove(bins_list[i])
            bad_by_bin = bad_by_bin2a
            not_monotone_count = not_monotone_count2a
            balance = balance_a
        # 同样地，满足下述2种情况时返回方案b：（1）方案b能减轻非单调性而方案a不能；（2）方案a和b都能减轻非单调性，但是方案b的样本均匀性优于方案a
        elif not_monotone_count2a >= not_monotone_count_current and not_monotone_count2b < not_monotone_count_current or \
                                        not_monotone_count2a < not_monotone_count_current and not_monotone_count2b < not_monotone_count_current and balance_a > balance_b:
            bins_list[i] = bins_list[i] + bins_list[i_next]
            bins_list.remove(bins_list[i_next])
            bad_by_bin = bad_by_bin2b
            not_monotone_count = not_monotone_count2b
            balance = balance_b
        #如果方案a和b都不能减轻非单调性，返回均匀性更优的合并方案
        else:
            if balance_a< balance_b:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
            else:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
        return {'bins_list': bins_list, 'bad_by_bin': bad_by_bin, 'not_monotone_count': not_monotone_count,
                'balance': balance}


    N = df.shape[0]
    [badrate_bin, bad_by_bin] = BinBadRate(df, col, target)
    bins = list(bad_by_bin[col])
    bins_list = [[i] for i in bins]
    badRate = sorted(badrate_bin.items(), key=lambda x: x[0])
    badRate = [i[1] for i in badRate]
    not_monotone_count, not_monotone_position = FeatureMonotone(badRate)['count_of_nonmonotone'], FeatureMonotone(badRate)['index_of_nonmonotone']
    #迭代地寻找最优合并方案，终止条件是:当前的坏样本率已经单调，或者当前只有2箱
    while (not_monotone_count > 0 and len(bins_list)>2):
        #当非单调的箱的个数超过1个时，每一次迭代中都尝试每一个箱的最优合并方案
        all_possible_merging = []
        for i in not_monotone_position:
            merge_adjacent_rows = Merge_adjacent_Rows(i, np.mat(bad_by_bin), bins_list, not_monotone_count)
            all_possible_merging.append(merge_adjacent_rows)
        balance_list = [i['balance'] for i in all_possible_merging]
        not_monotone_count_new = [i['not_monotone_count'] for i in all_possible_merging]
        #如果所有的合并方案都不能减轻当前的非单调性，就选择更加均匀的合并方案
        if min(not_monotone_count_new) >= not_monotone_count:
            best_merging_position = balance_list.index(min(balance_list))
        #如果有多个合并方案都能减轻当前的非单调性，也选择更加均匀的合并方案
        else:
            better_merging_index = [i for i in range(len(not_monotone_count_new)) if not_monotone_count_new[i] < not_monotone_count]
            better_balance = [balance_list[i] for i in better_merging_index]
            best_balance_index = better_balance.index(min(better_balance))
            best_merging_position = better_merging_index[best_balance_index]
        bins_list = all_possible_merging[best_merging_position]['bins_list']
        bad_by_bin = all_possible_merging[best_merging_position]['bad_by_bin']
        not_monotone_count = all_possible_merging[best_merging_position]['not_monotone_count']
        not_monotone_position = FeatureMonotone(bad_by_bin[:, 3])['index_of_nonmonotone']
    return bins_list


def draw_IV(IV_dict,path, xlabel=None,figsize=(15,7),is_save=False):
    """
    信息值IV柱状图
    ---------------------
    param
    IV_dict: dict IV值字典
    path: str 文件存储地址
    xlabel: list x轴标签
    figsize: tupe 图片大小
    _____________________
    return
    draw_iv
    """
    IV_dict_sorted = sorted(IV_dict.items(),key=lambda x:x[1],reverse=True)
    ivlist= [i[1] for i in IV_dict_sorted]
    index= [i[0] for i in IV_dict_sorted]
    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index))+1
    ax1.bar(x, ivlist, width=0.5)#生成柱状图
    ax1.set_xticks(x)
    if xlabel:
        ax1.set_xticklabels(index, rotation=0, fontsize=8)

    ax1.set_ylabel('IV(Information Value)', fontsize=14)
    #在柱状图上添加数字标签
    for a, b in zip(x, ivlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)

    if is_save:
        plt.savefig(path+"higt_iv.png")
    plt.show()

def get_corr(data,figsize):
    """
    特征相关系数
    ------------------------
    parameter:
    data_new: dataFrame,columns must be number
    figsize: tupe,two number
    return:
            heatmap
    """
    #相关系数分析
    colormap = plt.cm.viridis
    plt.figure(figsize=figsize)
    plt.title('皮尔森相关性系数', y=1.05, size=8)
    mask = np.zeros_like(data.corr(),dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True,mask=mask)
    plt.savefig("feature_corr.png")
    plt.show()

def search_parma(train_X,train_y,estimator,param_grid,scoring="roc_auc",cv=5,
                 model_path="model.pkl",param_path="param.pkl"):
    """
    模型参数检索
    :param train_X: dataframe or arrary 训练集
    :param train_y:Series or array 目标变量
    :param estimator: estimator 输入模型,可以是单个模型,也可以是流水模型
    :param param_grid: dict 检索参数
    :param scoring: str 评价指标,默认auc
    :param cv: 交叉验证折数 默认5折
    :param model_path: 保存模型
    :param param_path: 保存模型参数
    :return: model
    """
    gs = GridSearchCV(estimator=estimator,param_grid=param_grid,scoring=scoring,cv=cv)
    print("starting fit model")
    gs.fit(X=train_X,y=train_y)
    print("best param:",gs.best_params_)
    print("best score:",gs.best_score_)
    model = gs.best_estimator_
    #savel
    load_pkl(filename=model_path,model="wb+",data=model)
    load_pkl(filename=param_path,model="wb+",data=gs.best_params_)
    return model,gs.best_params_

def Prob2Score(prob, basePoint=600,PDO=50):
    #将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    y2 = basePoint+PDO/np.log(2)*(-y)
    score = y2.astype("int")
    return score




