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
from catboost import CatBoostClassifier,Pool,cv 
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score,roc_curve,auc,f1_score,RocCurveDisplay
from scipy.stats import chi2_contingency,pointbiserialr
from scipy.stats.contingency import association
import joblib
#from google.colab import drive
#drive.mount('/content/drive')


data=pd.read_csv(r'C:\Users\Madhavi_J\OneDrive - Dell Technologies\My_data\Madhavi\Madhavi_downloads\LBDSC24042022-20220918T125555Z-001\LBDSC24042022\Project Session Track-2022-20230901T113943Z-001\Data.csv')

#Observation:
# we we have total 12684 rows and 25 columns 
# out of 25, we have 8 are numeric and 17 are categorical

data=data.rename(columns = { 'Accept(Y/N?)' : 'target'})


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

print('Duplicate records are removed!!')

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

print('Missing Value Treatment Done!!')


numeric_cols = data.select_dtypes(include = np.number) ### selects numeric columns
#numeric_cols.select_dtypes('int64').nunique()


#Observation
# the column toCoupon_GEQ5min shows single value and no varience. Hence its not significant.
# we will drop the toCoupon_GEQ5min column

data.drop(['toCoupon_GEQ5min'], axis=1, inplace=True)
numeric_cols = data.select_dtypes(include = np.number) ### selects numeric columns
cat_cols = data.select_dtypes(include = np.object) ### selects numeric columns

#Data Preprocessing
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
#print(corr)

# Observation
# from the chi2 correlation , we can see that  direction_same,direction_opp are not 
#correlated with Target veriable so lets drop them
data.drop(['direction_opp','direction_same'], axis=1, inplace=True)   
#data.groupby(['occupation', 'target']).size().unstack('target').sort_values(by=1, ascending=False)
df=data.copy()

#Feature Engineering###
#observation 
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
#print('Unique values:',df['occupation_class'].unique())
#print('-'*50)
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
#print('Unique values:',df['to_Coupon'].unique())
#print('-'*50)
df['to_Coupon'].describe()

print('Feature Engineering Done!!')



# lets drop occupation column as we have new column occupation_class, toCoupon_GEQ15min', 'toCoupon_GEQ15min as we have merged them
df.drop(['occupation','toCoupon_GEQ15min', 'toCoupon_GEQ15min'],axis=1, inplace=True)
print(df.columns)


#df.target.value_counts()
#px.bar(df,x=df.target.value_counts().index , y=df.target.value_counts().values , title= 'Target Disribution' , width = 600, height = 600,color=df.target.value_counts().index )
# Observation
# distribution is not equal but it does not indicate class imbalance 

#Feature Target
x=df.drop('target',axis=1)
y=df.target

#train_test_split
x_train,x_test, y_train,  y_test=train_test_split(x, y , test_size=0.2 , random_state=2, stratify= y)

#encoding
#df = pd.get_dummies(df, columns=['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50'] , drop_first=True) # use drop_first= True

#fit
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse = False,drop='if_binary')
ohe.fit(x_train[['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50'] ])

joblib.dump(ohe, 'OneHotEncoding.pkl')

#transform test
train_df= pd.DataFrame(ohe.transform(x_train[['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50']]), columns = list(ohe.get_feature_names_out()))
x_train=x_train.drop(['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50'], axis=1)
x_train=pd.concat([x_train.reset_index(drop=True), train_df.reset_index(drop=True)] , axis=1)
print(x_train.shape)

#transform test
test_df= pd.DataFrame(ohe.transform(x_test[['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50']]), columns = list(ohe.get_feature_names_out()))
x_test=x_test.drop(['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50'], axis=1)
x_test=pd.concat([x_test.reset_index(drop=True), test_df.reset_index(drop=True)] , axis=1)
print(x_test.shape)

print('Encoding Done!!')
# # Model Building 
##CatBoost 
#Catboost on encoded data
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
'verbose':0,
'allow_writing_files':False,
'train_dir':None
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
print(CatBoost_Classifier_Result)

joblib.dump(clf_ctb,'model.pkl')


