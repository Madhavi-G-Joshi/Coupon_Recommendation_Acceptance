#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
sns.set_style("whitegrid")

import os,sys
import pickle
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore'
                       )
os.getcwd()

import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as po
from plotly.subplots import make_subplots
import scipy
import datetime
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier,VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,Pool,cv 
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score,roc_curve,auc,f1_score,RocCurveDisplay
from scipy.stats import chi2_contingency,pointbiserialr
from scipy.stats.contingency import association

data=pd.read_csv('Data.csv')

#Observation:
# we we have total 12684 rows and 25 columns 
# out of 25, we have 8 are numeric and 17 are categorical

data=data.rename(columns = { 'Accept(Y/N?)' : 'target'})
cols=data.columns

for col in cols:
    print( col, data[col].unique() )

# ## Preprocessing & EDA
# 
# #handling missing data
# #handling outliers - As this is classification model, outlier treatment is not require
# #feature scaling - not required
# #encoding - convert text to numbers required 
# #handling imbalance 


# Remove duplicates
duplicate = data[data.duplicated(keep = 'last')]
# duplicate.shape #(74, 26)
data = data.drop_duplicates()


# Handling missing data
#data.isna().sum()
#data.isnull().sum()/len(data)*100

# missing values
print('Is there any missing value present or not?',data.isnull().values.any())
missing_percentage = data.isnull().sum()*100/len(data)
missing_value_df = pd.DataFrame({'missing_count': data.isnull().sum(),'missing_percentage': missing_percentage})
missing_value_df[missing_value_df.missing_count != 0]

#Observation:
#Car has 99% data missing hence we are going to drop the variable.
#We will use mode imputation the remaining null values in variables #car,CoffeeHouse,CarryAway,
#RestaurantLessThan20,Restaurant20To50         
data=data.drop(['car'], axis=1)
data['Bar'] = data['Bar'].fillna(data['Bar'].value_counts().index[0])
data['CoffeeHouse'] = data['CoffeeHouse'].fillna(data['CoffeeHouse'].value_counts().index[0])
data['CarryAway'] = data['CarryAway'].fillna(data['CarryAway'].value_counts().index[0])
data['RestaurantLessThan20'] = data['RestaurantLessThan20'].fillna(data['RestaurantLessThan20'].value_counts().index[0])
data['Restaurant20To50'] = data['Restaurant20To50'].fillna(data['Restaurant20To50'].value_counts().index[0])


numeric_cols = data.select_dtypes(include = np.number) ### selects numeric columns
#numeric_cols.select_dtypes('int64').nunique()


#Observation
# the column toCoupon_GEQ5min shows single value and no varience. Hence its not significant.
# we will drop the toCoupon_GEQ5min column

data.drop(['toCoupon_GEQ5min'], axis=1, inplace=True)
numeric_cols = data.select_dtypes(include = np.number) ### selects numeric columns



#As we can the numeric columns are also categories
cat_cols = data.select_dtypes(include = np.object) ### selects numeric columns
cat_cols


# # Correlation Analysis


summary_table = PrettyTable(["Column", "Correlated?", "P-val"]) #heading

def check_correlation(df,column,target_col):
    contigency = pd.crosstab(df[col],df[target_col])
    res = chi2_contingency(contigency)
    res_corr='Correlated 'if res[1] < 0.05  else 'Not Correlated'
    p_val=res[1]
    return column,target_col, res_corr,p_val
 
for col in data.columns:
    column,target_col, res_corr,p_val= check_correlation(data,col,'target')
    summary_table.add_row([column, res_corr,p_val])

table = pd.read_html(summary_table.get_html_string())
corr = table[0]
print(corr)

# Observation
# from the chi2 correlation , we can see that  direction_same,direction_opp are not 
#correlated with Target veriable so lets drop them
data.drop(['direction_opp','direction_same'], axis=1, inplace=True)

 

   
#data.groupby(['occupation', 'target']).size().unstack('target').sort_values(by=1, ascending=False)
df=data.copy()


#observation 
# 
# occupation feature has 25 no of distinct values, which creates very sparsity in data after Encoding. 
#Hence first,based on target and total count we will divide categories in classes 
# occupation_class where categorize all occupation in its suitable class.
occupation_dict = {'Healthcare Support':'High_Acceptance','Construction & Extraction':'High_Acceptance','Healthcare Practitioners & Technical':'High_Acceptance',
                   'Protective Service':'High_Acceptance','Architecture & Engineering':'High_Acceptance','Production Occupations':'Medium_High_Acceptance',
                    'Student':'Medium_High_Acceptance','Office & Administrative Support':'Medium_High_Acceptance','Transportation & Material Moving':'Medium_High_Acceptance',
                    'Building & Grounds Cleaning & Maintenance':'Medium_High_Acceptance','Management':'Medium_Acceptance','Food Preparation & Serving Related':'Medium_Acceptance',
                   'Life Physical Social Science':'Medium_Acceptance','Business & Financial':'Medium_Acceptance','Computer & Mathematical':'Medium_Acceptance',
                    'Sales & Related':'Medium_Low_Acceptance','Personal Care & Service':'Medium_Low_Acceptance','Unemployed':'Medium_Low_Acceptance',
                   'Farming Fishing & Forestry':'Medium_Low_Acceptance','Installation Maintenance & Repair':'Medium_Low_Acceptance','Education&Training&Library':'Low_Acceptance',
                    'Arts Design Entertainment Sports & Media':'Low_Acceptance','Community & Social Services':'Low_Acceptance','Legal':'Low_Acceptance','Retired':'Low_Acceptance'}
# occupation_dict
df['occupation_class'] = df['occupation'].map(occupation_dict)
print('Unique values:',df['occupation_class'].unique())
print('-'*50)
df['occupation_class'].describe()


# FE -- to_Coupon is combination of two features, toCoupon_GEQ15min and toCoupon_GEQ25min
to_Coupon = []
for i in range(df.shape[0]):
    if (list(df['toCoupon_GEQ15min'])[i] == 0):
        to_Coupon.append(0)
    elif (list(df['toCoupon_GEQ15min'])[i] == 1)and(list(df['toCoupon_GEQ25min'])[i] == 0):
        to_Coupon.append(1)
    else:
        to_Coupon.append(2)
        
df['to_Coupon'] = to_Coupon
print('Unique values:',df['to_Coupon'].unique())
print('-'*50)
df['to_Coupon'].describe()



# lets drop occupation column as we have new column occupation_class, toCoupon_GEQ15min', 'toCoupon_GEQ15min as we have merged them
df.drop(['occupation','toCoupon_GEQ15min', 'toCoupon_GEQ15min'],axis=1, inplace=True)
df.columns


# # Feature Selection

numeric_cols = df.select_dtypes(include = np.number) ### selects numeric columns
#numeric_cols

from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
for col in numeric_cols.columns:
    df[col]=normalizer.fit_transform(df[col].values.reshape(-1,1))

data_fe=df.copy()

# # Encoding

df = pd.get_dummies(df, columns=['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50'] , drop_first=True) # use drop_first= True
df= df.rename(columns = {'passanger_Friend(s)' : 'passanger_Friends' , 'passanger_Kid(s)': 'passanger_Kids',
'coupon_Carry out & Take away' :'coupon_CarryOut_TakeAway', 'coupon_Restaurant(20-50)':'coupon_Restaurant_20to50',
  'coupon_Restaurant(<20)': 'coupon_Restaurant_LessThan20',  'education_Graduate degree (Masters or Doctorate)' :  'education_Graduate degree_masters',
  'income_$12500 - $24999' : 'income_12500to24999_usd','income_$25000 - $37499' :'income_25000to37499_usd', 
  'income_$37500 - $49999' :'income_37500to49999_usd','income_$50000 - $62499':'income_50000to62499_usd', 
  'income_$62500 - $74999':'income_62500to74999_usd','income_$75000 - $87499':'income_75000to87499_usd',
  'income_$87500 - $99999': 'income_87500to99999_usd','income_Less than $12500':'income_Lessthan12500',
        'Bar_4~8':'Bar_4to8','CoffeeHouse_4~8':'CoffeeHouse_4to8','CarryAway_4~8':'CarryAway_4to8',
          'RestaurantLessThan20_4~8':'RestaurantLessThan20_4to8',
          'Restaurant20To50_4~8': 'Restaurant20To50_4to8'})
print(df.columns)

df.target.value_counts()
px.bar(df,x=df.target.value_counts().index , y=df.target.value_counts().values , title= 'Target Disribution' , width = 600, height = 600,color=df.target.value_counts().index )
# Observation
# distribution is not equal but it does not indicate class imbalance 


# # Train Test Split
x=df.drop('target',axis=1)
y=df.target

#train_test_split
x_train,x_test, y_train,  y_test=train_test_split(x, y , test_size=0.2 , random_state=2, stratify= y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)





# # Model Building 

# # Logistic Regression
clf_lr = LogisticRegression(random_state=0,C=100).fit(x_train, y_train)
y_pred_train_lr= clf_lr.predict(x_train)
y_pred_test_lr= clf_lr.predict(x_test)

Train_AUC = roc_auc_score(y_train,clf_lr.predict_proba(x_train)[:,1])
Train_accuracy = accuracy_score(y_train,y_pred_train_lr)
Train_f1 = f1_score(y_train,y_pred_train_lr)

Test_AUC = roc_auc_score(y_test,clf_lr.predict_proba(x_test)[:,1])
Test_accuracy = accuracy_score(y_test,y_pred_test_lr)
Test_f1 = f1_score(y_test,y_pred_test_lr)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" , "Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["Logistic Regression",round(Train_AUC,3),round(Train_accuracy, 3), round(Train_f1,3),round(Test_AUC,3),round(Test_accuracy,3), round(Test_f1,3) ])
table = pd.read_html(summary_table.get_html_string())
Logistic_Regression_Result = table[0]
#Logistic_Regression_Result


# ## KNN
clf = KNeighborsClassifier(n_neighbors=21).fit(x_train, y_train)
y_pred_train_knn= clf.predict(x_train)
y_pred_test_knn= clf.predict(x_test)

Train_AUC_knn = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
Train_accuracy_knn = accuracy_score(y_train,y_pred_train_knn)
Train_f1_knn = f1_score(y_train,y_pred_train_knn)

Test_AUC_knn = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])
Test_accuracy_knn = accuracy_score(y_test,y_pred_test_knn)
Test_f1_knn = f1_score(y_test,y_pred_test_knn)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" , "Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["K-Nearest Neighbor",round(Train_AUC_knn,3),round(Train_accuracy_knn, 3), round(Train_f1_knn,3),round(Test_AUC_knn,3),round(Test_accuracy_knn,3), round(Test_f1_knn,3) ])
table = pd.read_html(summary_table.get_html_string())
K_Nearest_Neighbor_Result = table[0]
#K_Nearest_Neighbor_Result

# # Decision Tree
clf_dt = DecisionTreeClassifier( max_depth=10 , min_samples_split= 10, random_state=101)
clf_dt.fit(x_train,y_train)
y_pred_train_dt= clf_dt.predict(x_train)
y_pred_test_dt= clf_dt.predict(x_test)

Train_AUC_dt = roc_auc_score(y_train,clf_dt.predict_proba(x_train)[:,1])
Train_accuracy_dt = accuracy_score(y_train,y_pred_train_dt)
Train_f1_dt = f1_score(y_train,y_pred_train_dt)

Test_AUC_dt = roc_auc_score(y_test,clf_dt.predict_proba(x_test)[:,1])
Test_accuracy_dt = accuracy_score(y_test,y_pred_test_dt)
Test_f1_dt = f1_score(y_test,y_pred_test_dt)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" , "Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["Decision Tree",round(Train_AUC_dt,3),round(Train_accuracy_dt, 3), round(Train_f1_dt,3),round(Test_AUC_dt,3),round(Test_accuracy_dt,3), round(Test_f1_dt,3) ])
table = pd.read_html(summary_table.get_html_string())
Decision_Tree_Result = table[0]
#Decision_Tree_Result

# # Support Vector Machine
clf_svc = SVC(kernel='rbf')
clf_svc.fit(x_train, y_train)
y_pred_train_svm= clf_svc.predict(x_train)
y_pred_test_svm= clf_svc.predict(x_test)

Train_AUC_svc = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
Train_accuracy_svc = accuracy_score(y_train,y_pred_train_svm)
Train_f1_svc = f1_score(y_train,y_pred_train_svm)

Test_AUC_svc = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])
Test_accuracy_svc = accuracy_score(y_test,y_pred_test_svm)
Test_f1_svc = f1_score(y_test,y_pred_test_svm)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading

#summary_table.add_row(["Support Vector Classification","Ordinal Encoding",best_C_OrEnc,'',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["SVM",round(Train_AUC_svc,3),round(Train_accuracy_svc, 3), round(Train_f1_svc,3),round(Test_AUC_svc,3),round(Test_accuracy_svc,3), round(Test_f1_svc,3) ])

table = pd.read_html(summary_table.get_html_string())
Support_Vector_Classifier_Result = table[0]
#Support_Vector_Classifier_Result

# SVM Cross Validation required
from sklearn.model_selection import cross_val_score
svm_train_accuracy = cross_val_score(clf_svc, x_train, y_train, cv=10)
svm_test_accuracy = cross_val_score(clf_svc, x_test, y_test, cv=10)
print(svm_train_accuracy)
print("***************"*5)
print('svm_ mean train accuracy', svm_train_accuracy.mean())
print("***************"*5)
print('svm_ max train accuracy', svm_train_accuracy.max())
print("***************"*5)
print(svm_test_accuracy)
print("***************"*5)
print('svm_ mean test accuracy',svm_test_accuracy.mean())
print("***************"*5)
print('svm_ max test accuracy',svm_test_accuracy.max())


# # Ensemble Methods
# ## Random Forest

from sklearn.model_selection import cross_val_score, GridSearchCV,RandomizedSearchCV


#Hyper parameter tuning

clf_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,max_features='log2',min_samples_leaf=3,random_state=42, n_jobs=-1)
parameters = {'max_depth':[10,20,50], 'n_estimators':[1000,2000],
              'max_features': ['auto', 'sqrt', 'log2', None]}
model = RandomizedSearchCV(clf_rf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
model.fit(x_train, y_train)

best_depth = model.best_params_['max_depth']
best_n_estimators = model.best_params_['n_estimators']

clf_rf = RandomForestClassifier(n_estimators=best_n_estimators,criterion='gini',max_depth=best_depth, max_features='log2', min_samples_leaf=3, random_state=0, n_jobs=-1)
clf_rf.fit(x_train, y_train)
y_pred_train_rf= clf_rf.predict(x_train)
y_pred_test_rf= clf_rf.predict(x_test)

Train_AUC_rf = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
Train_accuracy_rf = accuracy_score(y_train,y_pred_train_rf)
Train_f1_rf = f1_score(y_train,y_pred_train_rf)

Test_AUC_rf = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])
Test_accuracy_rf = accuracy_score(y_test,y_pred_test_rf)
Test_f1_rf = f1_score(y_test,y_pred_test_rf)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading

#summary_table.add_row(["Support Vector Classification","Ordinal Encoding",best_C_OrEnc,'',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["Random Forest",round(Train_AUC_rf,3),round(Train_accuracy_rf, 3), round(Train_f1_rf,3),round(Test_AUC_rf,3),round(Test_accuracy_rf,3), round(Test_f1_rf,3) ])

table = pd.read_html(summary_table.get_html_string())
Random_Forest_Classifier_Result = table[0]
#Random_Forest_Classifier_Result

# Cross Validation required
from sklearn.model_selection import cross_val_score
rf_train_accuracy = cross_val_score(clf_rf, x_train, y_train, cv=10)
rf_test_accuracy = cross_val_score(clf_rf, x_test, y_test, cv=10)
print(rf_train_accuracy)
print("***************"*5)
print('Random Forest mean train accuracy', rf_train_accuracy.mean())
print("***************"*5)
print('Random Forest max train accuracy', rf_train_accuracy.max())
print("***************"*5)
print(rf_test_accuracy)
print("***************"*5)
print('Random Forest mean test accuracy',rf_test_accuracy.mean())
print("***************"*5)
print('Random Forest max test accuracy',rf_test_accuracy.max())


# ##  extra_tree Classifier
clf_eta = ExtraTreesClassifier(n_estimators=1000,max_depth=50 ,random_state=0, n_jobs=-1)
clf_eta.fit(x_train, y_train)
y_pred_train_eta= clf_eta.predict(x_train)
y_pred_text_eta= clf_eta.predict(x_test)

Train_AUC_eta = roc_auc_score(y_train,clf_eta.predict_proba(x_train)[:,1])
Train_accuracy_eta = accuracy_score(y_train,y_pred_train_eta)
Train_f1_eta = f1_score(y_train,y_pred_train_eta)

Test_AUC_eta = roc_auc_score(y_test,clf_eta.predict_proba(x_test)[:,1])
Test_accuracy_eta = accuracy_score(y_test,y_pred_text_eta)
Test_f1_eta = f1_score(y_test,y_pred_text_eta)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading

#summary_table.add_row(["Support Vector Classification","Ordinal Encoding",best_C_OrEnc,'',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["ExtraTree Classifier",round(Train_AUC_eta,3),round(Train_accuracy_eta, 3), round(Train_f1_eta,3),round(Test_AUC_eta,3),round(Test_accuracy_eta,3), round(Test_f1_eta,3) ])

table = pd.read_html(summary_table.get_html_string())
ExtraTree_Classifier_Result = table[0]
#ExtraTree_Classifier_Result

# ## Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf_adb = AdaBoostClassifier(estimator=DecisionTreeClassifier(),n_estimators=1000,learning_rate=0.1,random_state=42)
clf_adb.fit(x_train, y_train)
y_pred_train_adb= clf_adb.predict(x_train)
y_pred_test_adb= clf_adb.predict(x_test)


Train_AUC_adb = roc_auc_score(y_train,clf_adb.predict_proba(x_train)[:,1])
Train_accuracy_adb = accuracy_score(y_train,y_pred_train_adb)
Train_f1_adb = f1_score(y_train,y_pred_train_adb)

Test_AUC_adb = roc_auc_score(y_test,clf_adb.predict_proba(x_test)[:,1])
Test_accuracy_adb = accuracy_score(y_test,y_pred_test_adb)
Test_f1_adb = f1_score(y_test,y_pred_test_adb)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["AdaBoost Classifier",round(Train_AUC_adb,3),round(Train_accuracy_adb, 3), round(Train_f1_adb,3),round(Test_AUC_adb,3),round(Test_accuracy_adb,3), round(Test_f1_adb,3) ])
table = pd.read_html(summary_table.get_html_string())
AdaBoost_Classifier_Result = table[0]
#AdaBoost_Classifier_Result


# ## XGBoost

clf_xgb = XGBClassifier(random_state=100)
parameters = {'max_depth':[1, 5, 10, 50], 'n_estimators':[500,1000,2000]}
model = RandomizedSearchCV(clf_xgb, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
model.fit(x_train, y_train)
best_depth = model.best_params_['max_depth']
best_n_estimators = model.best_params_['n_estimators']

clf_xgb = XGBClassifier(max_depth=best_depth, n_estimators=best_n_estimators, random_state=100)
clf_xgb.fit(x_train, y_train)
y_pred_train_xgb= clf_xgb.predict(x_train)
y_pred_test_xgb= clf_xgb.predict(x_test)


Train_AUC_xgb = roc_auc_score(y_train,clf_xgb.predict_proba(x_train)[:,1])
Train_accuracy_xgb = accuracy_score(y_train,y_pred_train_xgb)
Train_f1_xgb = f1_score(y_train,y_pred_train_xgb)

Test_AUC_xgb = roc_auc_score(y_test,clf_xgb.predict_proba(x_test)[:,1])
Test_accuracy_xgb = accuracy_score(y_test,y_pred_test_xgb)
Test_f1_xgb = f1_score(y_test,y_pred_test_xgb)


summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading

#summary_table.add_row(["Support Vector Classification","Ordinal Encoding",best_C_OrEnc,'',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["XGBoost Classifier",round(Train_AUC_xgb,3),round(Train_accuracy_xgb, 3), round(Train_f1_xgb,3),round(Test_AUC_xgb,3),round(Test_accuracy_xgb,3), round(Test_f1_xgb,3) ])

table = pd.read_html(summary_table.get_html_string())
XGBoost_Classifier_Result = table[0]
#XGBoost_Classifier_Result

# Cross Validation required
from sklearn.model_selection import cross_val_score
xgb_train_accuracy = cross_val_score(clf_xgb, x_train, y_train, cv=10)
xgb_test_accuracy = cross_val_score(clf_xgb, x_test, y_test, cv=10)
print(xgb_train_accuracy)
print("***************"*5)
print('XGBoost mean train accuracy', xgb_train_accuracy.mean())
print("***************"*5)
print('XGBoost max train accuracy', xgb_train_accuracy.max())
print("***************"*5)
print(xgb_test_accuracy)
print("***************"*5)
print('XGBoost mean test accuracy',xgb_test_accuracy.mean())
print("***************"*5)
print('XGBoost max test accuracy',xgb_test_accuracy.max())


##CatBoost 

#dataset with no encoding 
x=data_fe.drop('target', axis=1)
y=data_fe.target
x_train_fe,x_test_fe ,  y_train_fe,y_test_fe=train_test_split(x,y, test_size=0.2, random_state=1)
cat_cols = list((x_train_fe.select_dtypes(include = np.object)).columns)
cat_cols.append('temperature')  #toCoupon_GEQ25min, to_Coupon, has_children

#get_ipython().system('pip install optuna')
import optuna
from optuna.integration import CatBoostPruningCallback
# ### Hyperparamter tuning using Optuna
def objective(trial: optuna.Trial) -> float:    

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "learning_rate":  trial.suggest_float('learning_rate', 0.001,1),
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = CatBoostClassifier(**param)

    pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    gbm.fit(
        x_train_fe,
        y_train_fe,
        eval_set=[(x_test_fe, y_test_fe)],
        cat_features=cat_cols,
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
        #random_seed=100
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    preds = gbm.predict(x_test_fe)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test_fe, pred_labels)

    return accuracy

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
study.optimize(objective, n_trials=2000, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# ### Catboost model on un-encoded data

import time
start = time.time()
pool_train = Pool(x_train_fe, y_train_fe,
                  cat_features = cat_cols)
pool_test = Pool(x_test_fe, cat_features =cat_cols)
params = {     'eval_metric': 'Accuracy',
 'iterations': 2000,
'random_seed': 42,
 'loss_function': 'Logloss',
'learning_rate':0.1
         }

clf_ctbfe = CatBoostClassifier(**params)

clf_ctbfe.fit(x_train_fe, y_train_fe , cat_features=cat_cols ,eval_set=(x_test_fe, y_test_fe),plot=True , verbose= 0)
y_pred_train_ctbfe= clf_ctbfe.predict(x_train_fe)
y_pred_test_ctbfe= clf_ctbfe.predict(x_test_fe)
end = time.time()
diff = end - start
print('Catboos_fe Execution time:', diff)

Train_AUC_ctbfe = roc_auc_score(y_train_fe,clf_ctbfe.predict_proba(x_train_fe)[:,1])
Train_accuracy_ctbfe = accuracy_score(y_train_fe,y_pred_train_ctbfe)
Train_f1_ctbfe = f1_score(y_train_fe,y_pred_train_ctbfe)

Test_AUC_ctbfe = roc_auc_score(y_test_fe,clf_ctbfe.predict_proba(x_test_fe)[:,1])
Test_accuracy_ctbfe = accuracy_score(y_test_fe,y_pred_test_ctbfe)
Test_f1_ctbfe = f1_score(y_test_fe,y_pred_test_ctbfe)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["CatBoost Classifier",round(Train_AUC_ctbfe,3),round(Train_accuracy_ctbfe, 3), round(Train_f1_ctbfe,3),round(Test_AUC_ctbfe,3),round(Test_accuracy_ctbfe,3), round(Test_f1_ctbfe,3) ])

table = pd.read_html(summary_table.get_html_string())
CatBoostfe_Classifier_Result = table[0]
#(CatBoostfe_Classifier_Result)





# ### Catboost on encoded data
import time
start = time.time()
clf_ctb = CatBoostClassifier()
#parameters = {'max_depth':[1, 5, 10, 50], 'n_estimators':[50,100,500,1000,2000]}
#model = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
#model.fit(x_train, y_train)
#best_n_estimators = model.best_params_['n_estimators']
params = {     'eval_metric': 'Accuracy',
 'iterations': 2500,
#'objective': 'Accuracy',
'random_seed': 0,
 'loss_function': 'Logloss',
'verbose':0
}

clf_ctb =CatBoostClassifier(**params)
clf_ctb.fit(x_train, y_train , eval_set=(x_test, y_test),plot=True , verbose= 0)
y_pred_train_ctb= clf_ctb.predict(x_train)
y_pred_test_ctb= clf_ctb.predict(x_test)
end = time.time()
diff = end - start
print('Execution time:', diff)
Train_AUC_ctb = roc_auc_score(y_train,clf_ctb.predict_proba(x_train)[:,1])
Train_accuracy_ctb = accuracy_score(y_train,y_pred_train_ctb)
Train_f1_ctb = f1_score(y_train,y_pred_train_ctb)

Test_AUC_ctb = roc_auc_score(y_test,clf_ctb.predict_proba(x_test)[:,1])
Test_accuracy_ctb = accuracy_score(y_test,y_pred_test_ctb)
Test_f1_ctb = f1_score(y_test,y_pred_test_ctb)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["CatBoost_OHE Classifier",round(Train_AUC_ctb,3),round(Train_accuracy_ctb, 3), round(Train_f1_ctb,3),round(Test_AUC_ctb,3),round(Test_accuracy_ctb,3), round(Test_f1_ctb,3) ])

table = pd.read_html(summary_table.get_html_string())
CatBoost_Classifier_Result = table[0]
# CatBoost_Classifier_Result
#clf_ctb.get_all_params()

# Cross Validation required
from sklearn.model_selection import cross_val_score
start=time.time()
catboost_train_accuracy = cross_val_score(clf_ctb, x_train, y_train, cv=10, verbose=0)
catboost_test_accuracy = cross_val_score(clf_ctb, x_test, y_test, cv=10, verbose=0)
end=time.time()
diff= end-start
print('Execution Time :', diff  )
print(catboost_train_accuracy)
print("***************"*5)
print('Catboost mean train accuracy', catboost_train_accuracy.mean())
print("***************"*5)
print('Catboost max train accuracy', catboost_train_accuracy.max())
print("***************"*5)
print(catboost_test_accuracy)
print("***************"*5)
print('Catboost mean test accuracy',catboost_test_accuracy.mean())
print("***************"*5)
print('Catboost max test accuracy',catboost_test_accuracy.max())


# ### Feature Importances using Shap values
#!pip install shap
import shap
shap.initjs()
explainer = shap.Explainer(clf_ctbfe)
shap_values = explainer.shap_values(x_test_fe)
shap.summary_plot(shap_values, x_test_fe)
shap.summary_plot(shap_values, features=x_test_fe, feature_names=x_test_fe.columns, plot_type='bar')
importances = pd.DataFrame (clf_ctbfe.get_feature_importance (prettified=True), columns = ["Feature Id", "Importances"])
#print (importances.head (20))
shap_values = clf_ctbfe.get_feature_importance (Pool (x_test_fe, y_test_fe,cat_cols), type = "ShapValues")
#print (shap_values)
shap_values = shap_values[:,: -1]
shap.summary_plot(shap_values,x_test_fe)
#shap.plots.bar(shap_values)

#fig=shap.summary_plot(shap_values, x_test_fe, show=False)

print("catboost model parameters", clf_ctbfe.get_params ())
print("catboost model seed", clf_ctbfe.random_seed_)
pd.DataFrame({'catboost feature_importance': clf_ctbfe.get_feature_importance(Pool (x_test_fe,y_test_fe, cat_cols)), 'feature_names': x_test_fe.columns}).sort_values(by=['feature_importance'],  ascending=False)


# ## Stacking Classifier
from sklearn.ensemble import StackingClassifier
params = {     'eval_metric': 'Accuracy',
 'iterations': 1000,
 'loss_function': 'Logloss',
}
estimators = [('XGB', XGBClassifier(max_depth=5, n_estimators=1000, random_state=42)),
            ('RF',RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=50,max_features='log2',min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ('SVC', SVC(C=1,kernel='rbf',class_weight='balanced',probability=True)),
             ('CatBoost',CatBoostClassifier(**params))]
        
clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf_stack.fit(x_train, y_train)
y_pred_train_stack= clf_stack.predict(x_train)
y_pred_test_stack= clf_stack.predict(x_test)

Train_AUC_stack = roc_auc_score(y_train,y_pred_train_stack)
Train_accuracy_stack = accuracy_score(y_train,y_pred_train_stack)
Train_f1_stack = f1_score(y_train,y_pred_train_stack)

Test_AUC_stack = roc_auc_score(y_test,y_pred_test_stack)
Test_accuracy_stack = accuracy_score(y_test,y_pred_test_stack)
Test_f1_stack = f1_score(y_test,y_pred_test_stack)

summary_table = PrettyTable(["Model", "Train_roc_auc_score", "Train_accuracy","Train_f1" ,"Test_roc_auc_score",'Test_accuracy' ,'Test_f1']) #heading
summary_table.add_row(["Stacking Classifier",round(Train_AUC_stack,3),round(Train_accuracy_stack, 3), round(Train_f1_stack,3),round(Test_AUC_stack,3),round(Test_accuracy_stack,3), round(Test_f1_stack,3) ])

table = pd.read_html(summary_table.get_html_string())
Stacking_Classifier_Result = table[0]
#Stacking_Classifier_Result


# # Model Comparison

Model_Result = [Logistic_Regression_Result,K_Nearest_Neighbor_Result,Decision_Tree_Result,Support_Vector_Classifier_Result,ExtraTree_Classifier_Result,
                AdaBoost_Classifier_Result,Random_Forest_Classifier_Result,XGBoost_Classifier_Result,
                CatBoost_Classifier_Result,CatBoostfe_Classifier_Result,Stacking_Classifier_Result] 
Result = pd.concat(Model_Result,ignore_index=True)

# Result.to_csv('/content/drive/MyDrive/Applied AI/CS1/Model_Result.csv')
(Result).sort_values(by=['Test_accuracy'],ascending=False).head(10)


# __Observations:__
# 
# * Stacking Classifier,Random Forest Classifier, XGB Classifier, Catboost Classifier these models perform best than other models.
# * The best-performed model is Catboost Classifier so far

# Lets evaluate catboost model results

fig, ax = plt.subplots(figsize =(10,5))
ax.grid(False)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test_fe, y_pred_test_ctbfe), display_labels=clf_ctbfe.classes_)
disp.plot(ax=ax)
plt.show()

fig, ax = plt.subplots(figsize =(10,5))
ax.grid(False)
disp=ConfusionMatrixDisplay(confusion_matrix(y_train, y_pred_train_stack), display_labels=clf_stack.classes_) 
disp.plot(ax=ax)
plt.show()


fig, ax = plt.subplots(figsize =(10,5))
ax.grid(False)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test_stack), display_labels=clf_stack.classes_) 
disp.plot(ax=ax)
plt.show()



'''roc_curve(
    y_true,
    y_score, # the probability score
    *,
    pos_label=None,
    sample_weight=None,
    drop_intermediate=True,
)'''

fpr, tpr, thresholds = roc_curve(y_train, y_pred_train_stack)# The confidence score for a sample is proportional to the signeddistance of that sample to the hyperplane.
roc_auc = auc(fpr, tpr)
#display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                  estimator_name='Catboost')
#display.plot()
#plt.title('ROC AUC Curve')
#plt.show()
fig = px.area(
    x=fpr, y=tpr,
    title=f'Train ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)


# In[ ]:


'''roc_curve(
    y_true,
    y_score, # the probability score
    *,
    pos_label=None,
    sample_weight=None,
    drop_intermediate=True,
)'''

fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_stack)# The confidence score for a sample is proportional to the signeddistance of that sample to the hyperplane.
roc_auc = auc(fpr, tpr)
#display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                  estimator_name='Catboost')
#display.plot()
#plt.title('ROC AUC Curve')
#plt.show()

fig = px.area(
    x=fpr, y=tpr,
    title=f'Test ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)


from pycaret.classification import *

# creating training dataset
dataset = df.sample(frac=0.90, random_state=1)
# test data
dataset_unseen = df.drop(dataset.index).reset_index(drop=True)

pyc_model = setup(data=dataset, target='target' , session_id=201)

#!pip install numba==0.56.3


compare_models()
catboost = create_model('catboost')
#catboost.get_all_params()
plot_model(catboost, plot='confusion_matrix')
plot_model(catboost, plot='auc')
predict_model(catboost)
unseen_predictions = predict_model(catboost, data= dataset_unseen)
print('Catboost Pycaret unseen_predictions', unseen_predictions)

# __Observation:__
# * we are having having train and test accuracy with Catboost model with manual as well as autoML Pycaret method
# * With Manual , we see catboost has highest train accuracy of 84.6% and test accuracy of 76.6%
# * with Pycaret , we see that catboost, we are getting train accuracy and test accuracy 76.4%
# 


