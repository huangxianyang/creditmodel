# -*- coding: utf-8 -*-
"""
评分卡示例:
0.数据探索分析,1.数据清洗,2.特征分箱,3.特征选择,4.模型训练,5.评分卡构建,6.模型评估,7.风险决策
"""
from sklearn.externals import joblib
import riskscore as rs

file = './result/'
# 导入数据
germancredit = rs.Germancredit()
german = germancredit.get_data()
print("数据描述:", germancredit.get_describe())
print("数据样例:", german.head())

# 预处理变量
########################################
# 0.数据探索分析
#######################################
bs = rs.BasicStat(df=german)
# 字段基本统计
for col in german.columns:
    describe = bs.feature_describe(x=col)
    print(describe)
# 数据分布报告
bs.df_report(filename=file+'germancredit.html')
# 分布图
plot = rs.Plot(df=german)
# 缺失可视化
plot.plot_miss(filename=file+'miss_draw.png',asc=1,figsize=(15,8))
# 单变量探索(示例)
plot.draw_histogram(var='credit_amount',num_bins=20,filename=file+'credit_amount.png')

########################################
# 1.数据清洗
#######################################
pr = rs.Preprocess()
# 变量类型划分(实际业务中不能使用此方法)
# num_col,cat_col = pr.split_col(df=german,no_split=['creditability'],min_size=4)
# print("number:",num_col,"\n","category:",cat_col)
# 数值化
for col in germancredit.sub_col:
    german[col] = german[col].apply(lambda x:int(str(x)[1:]))
# 替换
german = pr.replace_str(df=german,replace_dict=germancredit.rep_dict)
# 连续变量中特殊字符检测
str_set = pr.find_str(df=german,num_col=germancredit.num_col)
print("特殊字符:",str_set)
# 异常字符检测
special_set = pr.special_char(df=german,feature_col=germancredit.all_feature)
print("异常字符:",special_set)
# 异常值检测及处理
german = pr.outlier(df=german,col="age_in_years",low_percent=0.05,up_percent=0.97,cat_percent=0.001,special_value=None)
# 删除重复行
german = pr.drop_dupl(df=german,axis=0)
# 缺失和众数删除
delete_col = pr.drop_nan_mode(df=german,nan_percent=0.9,mode_percent=0.95,col_list=germancredit.all_feature,drop=False)
print("删除变量:",delete_col)
all_feature = [i for i in germancredit.all_feature if i not in delete_col]
num_col = [i for i in germancredit.num_col if i not in delete_col]
cat_col = [i for i in germancredit.cat_col if i not in delete_col]
int_col = [i for i in germancredit.int_col if i not in delete_col]
german = german[all_feature+['target']]
# 缺失填充
# 类别
german = pr.fill_nan(df=german,value='-9999',method='mode',col_list=cat_col,min_miss_rate=0.05)
# 数值
german = pr.fill_nan(df=german,value=-9999,method='mode',col_list=num_col,min_miss_rate=0.05)
# 保存清洗后的数据
german['target'] = german['target'].apply(lambda x: 1 if x ==2 else 0)

# 过采样,注意: 随机采样,平衡采样方法无需数值化, 过采样和下采样方法必须数值化
df,factorize_dict = pr.factor_map(df=german,col_list=cat_col)
ds = rs.DataSample(df=df,target='target')
df_res = ds.over_sample(method='BorderLine',sampling_strategy="minority")
# 变量值映射,对已数值化变量映射为字符型
# 类别变量和有序离散变量整数化
for col in int_col:
    df_res[col] = df_res[col].apply(lambda x:int(x))
    df[col] = df[col].apply(lambda x:int(x))

replace_dict={k:{i:j for j,i in v.items()} for k,v in factorize_dict.items()}
# print(replace_dict)
df_res = pr.replace_str(df=df_res,replace_dict=replace_dict)
df = pr.replace_str(df=df,replace_dict=replace_dict)
#############################################
# 2.特征工程
#############################################
fe = rs.FeatureBin(df=df_res,target="target",special_values=['-9999',-9999],
                   min_per_fine_bin=0.01,stop_limit=0.01,min_per_coarse_bin=0.05,max_num_bin=8,method="tree")
# 类别变量分箱
cat_bin_dict,cat_var_iv = fe.category_bin(bin_feature=cat_col,max_num_bin=5)
# 数值变量分箱
num_bin_dict,num_var_iv = fe.number_bin(bin_feature=num_col,max_num_bin=5,no_mono_feature=None)
# 自定义分箱
# self_bin_dict,self_var_iv = fe.self_bin(var='',special_values=[],breaks_list={})
# 合并分箱结果
bin_dict = {**cat_bin_dict,**num_bin_dict}
var_iv = {**cat_var_iv,**num_var_iv}
# 分箱可视化
bplt = rs.WoeBinPlot(bin_dict=bin_dict)
_ = bplt.woe_plot(features=all_feature,show_iv=True,save=True,path=file)
# woe 转换
df_woe,woe_feature = rs.woe_trans(df=df_res,bin_dict=bin_dict,trans_feature=all_feature,target="target")

###############################################
# 3.特征选择
##############################################
sf = rs.SelectFeature(df_woe=df_woe)
# 基于特征IV
high_iv = sf.baseOn_iv(var_iv=var_iv,threshold=0.02,is_save=False,path=file)
high_iv = {k+"_woe":v for k,v in high_iv.items()}
print("high iv feature:",high_iv)
# 基于特征重要度
high_importance = sf.baseOn_importance(features=woe_feature,target="target",n_estimators=100,is_save=False,path=file)
print("high importance feature:",high_importance)
# 基于共线性检验
feature_vif1 = sf.baseOn_collinear(features=high_iv,cor_threshold=0.8,is_save=False,path=file)
print("两两共线性检验:",feature_vif1)
feature_vif2 = sf.baseOn_vif(features=high_iv,max_vif=10)
print("VIF共线性检验:",feature_vif2)
# 基于逐步回归
step_feature = sf.baseOn_steplr(features=feature_vif2,target='target',C=1,class_weight='balanced',norm="AIC")
# 基于l1正则化
select_feature = sf.baseOn_l1(features=list(step_feature.keys()),target='target',C=0.1,class_weight="balanced",drop_plus=False)
print("基于L1正在:",select_feature)

##############################################
# 4.模型训练
#############################################
select_feature = list(select_feature.keys())
# 超参优化
ps = rs.SearchParam(X=df_woe[select_feature].values,y=df_woe['target'].values)
grid_param = ps.grid_search(param_grid={"penalty":["l1","l2"],"C":[0.01,0.05,0.1,0.5,1]},cv=5,class_weight='balanced',scoring='roc_auc')
print("grid best param:",grid_param)
bayes_param = ps.bayes_search(param_grid={"penalty":["l1","l2"],"C":(0.001,1)},cv=5,n_iter=10,class_weight='balanced',scoring='roc_auc')
print("bayes best param:",bayes_param)
# 模型实例化
tlr = rs.TrainLr(df_woe=df_woe,features=select_feature,target='target',penalty='l2',class_weight='balanced')
lr = tlr.lr(C=0.37,filename=file)
lr_cv = tlr.lr_cv(Cs=[0.01,0.05,0.1,0.5,1],cv=5,scoring='roc_auc',filename=file)

################################
# 5.评分卡构建
################################
model_feature = [i.replace("_woe","") for i in select_feature]
sc = rs.ScoreCard(lr=lr_cv,bin_dict=bin_dict,model_feature=model_feature,score0=600,pdo=50)
# 输出标准评分卡
df_card = sc.score_card(return_df=True)
df_card.to_excel(file+"score_card.xlsx")

dict_card = sc.score_card(return_df=False)
joblib.dump(dict_card,file+"score_card.pkl")
df_res['score'] = sc.score_ply(df=df_res)

###############################
# 6.模型评估,真实样本评估
##############################
german_woe,_ = rs.woe_trans(df=df,bin_dict=bin_dict,trans_feature=all_feature,target="target")
# print(german_woe.isnull().sum())
y_prob = lr_cv.predict_proba(german_woe[select_feature].values)[:,1]
y_true = german_woe['target'].values
# 模型指标
model_norm = rs.model_norm(y_true=y_true,y_prob=y_prob)
print("模型测试结果: ",model_norm)
# 作图
mplt = rs.PlotModel(y_true=y_true,y_prob=y_prob)
mplt.plot_roc_curve(filename=file)
mplt.plot_ks_curve(filename=file)
mplt.plot_confusion_matrix(labels=[0,1],filename=file)
df['score'] = sc.score_ply(df=df,only_total_score=True)
###############################
# 7.风险决策, score cut_off
##############################
cut_off_score = rs.stragety_score(score_df=df,step=25,score="score",label='target',
                                  amount=5000,tenor=6,IRR=0.3,capital_cost=0.08,
                                  guest_cost=100,data_cost=30,bad_loss=0.6)
cut_off_score.to_excel(file+"cut_off_score.xlsx")