#-*- utf-8 -*-
#model training and model evaluating
#Modeling.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import function.utils as F
from sklearn.metrics import roc_curve,f1_score,recall_score,confusion_matrix,auc,classification_report
import matplotlib.pyplot as plt
import itertools

def plot_roc_curve(pred_y, y):
    """
    plot roc curve
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    ----------------------------------
    plt object and auc and ks value
    """
    fpr, tpr, _ = roc_curve(y, pred_y)
    c_stats = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label="ROC curve")
    auc_value = "AUC = %.3f" % c_stats
    ks_value = "KS = %.3f" % max(abs(tpr - fpr))
    plt.text(0.8, 0.4, ks_value, bbox=dict(facecolor="g", alpha=0.7))
    plt.text(0.8, 0.2, auc_value, bbox=dict(facecolor='r', alpha=0.5))
    plt.xlabel('False positive rate') #假正率
    plt.ylabel('True positive rate') #真正率
    plt.title('ROC curve')  # ROC 曲线
    plt.legend(loc='best')
    plt.savefig("ROC_draw.png")
    plt.show()

    return auc_value


def best_threshold(pred_y,y):
    """
    cut best pred
    :param pred_y: y of prediction
    :param y: real y
    :return: ks_value and draw ks
    """
    fpr, tpr, thr = roc_curve(y, pred_y)
    max_ks = 0
    best_thr = 0.5
    for i in range(len(thr)):
        if abs(fpr[i] - tpr[i]) > max_ks:
            max_ks = abs(fpr[i] - tpr[i])
            best_thr = thr[i]
    return best_thr


def plot_ks_curve(pred_y,y):
    """
    plot ks curve
    :param pred_y: y of prediction
    :param y: real y
    :return: ks_value and draw ks
    """
    fpr, tpr, thr = roc_curve(y, pred_y) #假正率 真正率 概率阈值
    ks = abs(fpr-tpr) #ks 序列
    ks_value = "KS = %.3f" % max(ks) #ks值
    ks_thr = best_threshold(pred_y,y) #最佳切分概率
    plt.plot(thr,fpr,label='cum_good', color='blue', linestyle='-', linewidth=2) #假正率 累计好
    plt.plot(thr,tpr,label='cum_bad', color='red', linestyle='-', linewidth=2) #真正率,累计坏
    plt.plot(thr, ks, label='ks',color='green', linestyle='-', linewidth=2) #ks曲线
    plt.axvline(ks_thr, color='gray', linestyle='--') #最佳切分概率直线
    plt.axhline(ks_value, color='green', linestyle='--') #ks值直线
    plt.title('KS=%s ' %np.round(ks_value, 3), fontsize=15)
    plt.savefig("ks_curve.png") #保存
    plt.show()
    return ks_value

def plot_confusion_matrix(pred_y,y,labels,normalize=False,cmap=plt.cm.Blues):
    """
    混淆矩阵
    ------------------------------------------
    Params
    pred_y: predict results
    y：real data labels
    labels: list, labels class
    normalize: bool, True means trans results to percent
    cmap: color index
    ------------------------------------------
    Return
    plt object
    """
    cm = confusion_matrix(y, pred_y, labels=labels)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  #在指定的轴上展示图像

    plt.colorbar()  # 增加色柱
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)  # 设置坐标轴标签
    plt.yticks(tick_marks, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title("confusion matrix")
    plt.savefig("confunsion_matrix.png")
    plt.show()


def class_report(y_pred,y,lables=None,target_names=None):
    """
    类别报告
    :param y_pred: like list predict y
    :param y: like list ture y
    :param lables: list 标签
    :param target_names: list 标签名称
    :return: report and f1 and recall
    """
    f1 = f1_score(y_true=y,y_pred=y_pred,labels=lables)
    recall = recall_score(y_true=y,y_pred=y_pred,labels=lables)
    report = classification_report(y_true=y,y_pred=y_pred,labels=lables,target_names=target_names)
    return f1,recall,report

class ScoreCardModel(LogisticRegression):
    """
    评分卡模型
    """
    def search_lr_param(self,X, y,C,cv=5,penalty="l2",class_weight="balanced",scoring="roc_auc"):
        """
        逻辑回归最优参数检索,返回正则参数及正则力度参数
        :param X: trainDataSet of X, all feature values must be number or bool
        :param y: trainDataSet of lable,label must be number or bool
        :param C: list
        :param cv: int
        :param penalty: str, l1, l2
        :param class_weight: dict or str
        :param scoring: str
        :return: param
        """
        param_grid = {"penalty":penalty,"C":C} #参数设置
        lg = LogisticRegression(class_weight=class_weight) #初始化模型
        grid_search = GridSearchCV(lg, param_grid=param_grid, cv=cv, scoring=scoring) #网格参数搜索
        grid_search = grid_search.fit(X, y) #拟合
        return grid_search.best_params_

    def create_lr_model(self,X,y,feature_name,penalty="l2",C=0.1,class_weight="balanced"):
        """
        逻辑回归模型,评分卡模型准备
        :param X:trainDataSet of X, all feature values must be number or bool
        :param y:trainDataSet of lable,label must be number or bool
        :param feature_name: list 特征序列
        :param penalty:str default l2
        :param C: float default 0.1
        :param class_weight:str default balanced
        :return: save model and feature param and AUC and KS
        """
        # 建模
        LR = LogisticRegression(C=C, penalty=penalty, class_weight=class_weight)
        LR.fit(X, y)
        # 模型系数
        paramsEst = pd.Series(LR.coef_.tolist()[0], index=feature_name)
        paramsEst["intercept"] = LR.intercept_.tolist()[0]
        #print("模型系数:", paramsEst)
        #AUC and KS
        y_pred = LR.predict_proba(X)[:,1]
        auc_value, ks_value = plot_roc_curve(pred_y=y_pred,y=y)
        # save model
        F.load_pkl(filename="LR.pkl",model="wb+",data=LR)
        return paramsEst,auc_value,ks_value

    def model_evaluation(self,y_ture,y_pred):
        """
        模型评估
        :param y_ture: 真实值
        :param y_pred: 预测概率
        :return: 评估指标
        """
        #auc
        AUC = plot_roc_curve(pred_y=y_pred,y=y_ture)
        KS = plot_ks_curve(pred_y=y_pred,y=y_ture)
        best_thre = best_threshold(pred_y=y_pred,y=y_ture)
        y_predict = []
        for i in y_pred:
            if i >= best_thre:
                y_predict.append(1)
            else:
                y_predict.append(0)

        f1, recall, report = classification_report(y_true=y_ture,y_pred=np.array(y_predict),labels=[0,1],
                                                   target_names=["good","bad"])
        return AUC,KS,f1,recall,report,best_thre

    def scorecard(self,bins, model, model_var, points0=600, pdo=50):
        '''
        Creating a Scorecard
        ------
        `scorecard` creates a scorecard based on the results from `woebin`
        and LogisticRegression of sklearn.linear_model

        Params
        ------
        bins: Binning information generated from `woebin` function.
        model: A LogisticRegression model object.
        points0: Target points, default 600.
        pdo: Points to Double the Odds, default 50.
        model_var: in model_var

        Returns
        ------
        DataFrame
            scorecard dataframe
        '''
        import re
        # coefficients
        A = points0
        B = pdo / np.log(2)

        # bins # if (is.list(bins)) rbindlist(bins)
        if isinstance(bins, dict):
            bins = pd.concat(bins, ignore_index=True)
        xs = [re.sub('_woe$', '', i) for i in model_var]
        # coefficients
        coef_df = pd.Series(model.coef_[0], index=np.array(xs)).loc[lambda x: x != 0]  # .reset_index(drop=True)

        # scorecard
        len_x = len(coef_df)
        basepoints = A - B * model.intercept_[0]
        card = {}
        card['basepoints'] = pd.DataFrame({'variable': "basepoints", 'bin': "基础分", 'points': round(basepoints, 2)},
                                          index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable'] == i, ['variable', 'bin', 'woe']] \
                .assign(points=lambda x: round(-B * x['woe'] * coef_df[i], 2))[["variable", "bin", "points"]]
        return card

    def card_excel(self,bin_dict, LR, model_var, points0=600, pdo=50,is_save=False):
        """
         # 评分模型生成,并输出excel格式文件
        :param bins: bin_dict 特征组合结果
        :param LR: object 逻辑回归模型
        :param model_var: list 入模变量, 注意后缀必须为_woe格式
        :param points0: int 基础分 默认600
        :param pdo: int 倍率 默认50
        :param is_save: bool 是否保存
        :return: score_df
        """

        card = self.scorecard(bins=bin_dict, model=LR, model_var=model_var, points0=points0, pdo=pdo)
        # transfer df
        score_df = pd.DataFrame()
        for var in card.keys():
            var_df = card[var]
            score_df = pd.concat([score_df, var_df])
        score_df.set_index(["variable"], inplace=True)
        if is_save:
            score_df.to_excel("scorecard.xlsx")
        return score_df

    def save_score(y_ture, y_pred, result_path, basePoint=600, PDO=50):
        """
        保存结果
        :param y_ture: array or series真实值
        :param y_pred: array or series真实值
        :param basePoint: int 基础分,默认600分
        :param PDO: int 倍率 默认30
        :return: 返回保存结果
        """
        df_score = pd.DataFrame()
        temp_list = []
        for i in range(len(y_pred)):
            temp_list.append(F.Prob2Score(prob=y_pred[i], basePoint=basePoint, PDO=PDO))
        #保存
        df_score["score"] = temp_list
        df_score["target"] = y_ture
        df_score.to_excel(result_path, index=False)