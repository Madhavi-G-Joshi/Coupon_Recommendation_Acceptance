from flask import Flask, request, render_template
import pandas as pd
import joblib

#initialize the app
app=Flask(__name__)


# load the model
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
model=joblib.load('model.pkl')

def predict_data(data, model):
    
    
    #load train data for imputation
    train_data=pd.read_csv('Data.csv')
    
    #############################Remove duplicates##########################################
    data = data.drop_duplicates()
    print('\n Duplicate records removed')


    ################################## Handling missing data#############################################
    data=data.drop(['car'], axis=1)
    data['Bar'] = data['Bar'].fillna(train_data['Bar'].value_counts().index[0])
    data['CoffeeHouse'] = data['CoffeeHouse'].fillna(train_data['CoffeeHouse'].value_counts().index[0])
    data['CarryAway'] = data['CarryAway'].fillna(train_data['CarryAway'].value_counts().index[0])
    data['RestaurantLessThan20'] = data['RestaurantLessThan20'].fillna(train_data['RestaurantLessThan20'].value_counts().index[0])
    data['Restaurant20To50'] = data['Restaurant20To50'].fillna(train_data['Restaurant20To50'].value_counts().index[0])
    print('\n Missing values treatment completed')
    #Observation
    # the column toCoupon_GEQ5min shows single value and no varience. Hence its not significant.
    # we will drop the toCoupon_GEQ5min column
    data.drop(['toCoupon_GEQ5min'], axis=1, inplace=True)
    data.drop(['direction_opp','direction_same'], axis=1, inplace=True)   
    df=data.copy()

    ######################################Feature Engineering#######################################
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
    # lets drop occupation column as we have new column occupation_class, toCoupon_GEQ15min', 'toCoupon_GEQ15min as we have merged them
    df.drop(['occupation','toCoupon_GEQ15min', 'toCoupon_GEQ15min'],axis=1, inplace=True)
    print('\n feature Engoneering done')
    
    ##########################Encoding#########################################################
    from sklearn.preprocessing import OneHotEncoder
    ohe=joblib.load('OneHotEncoding.pkl') 
    #transform test
    test_df= pd.DataFrame(ohe.transform(df[['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50']]), columns = list(ohe.get_feature_names_out()))
    df=df.drop(['destination','passanger','weather', 'coupon' , 'occupation_class','expiration','gender' ,'age','maritalStatus', 'education' , 'income','Bar' , 'CoffeeHouse','CarryAway' , 'RestaurantLessThan20', 'Restaurant20To50'], axis=1)
    x_test=pd.concat([df.reset_index(drop=True), test_df.reset_index(drop=True)] , axis=1)
    print('\n Encoding done')

    #predictions
    y_pred= model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    #print(y_pred, y_pred_proba)
    return y_pred[0], y_pred_proba[0]


@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict' , methods=['post'])
def predict():
    to_predict_dict = request.form.to_dict()
    to_predict_df = pd.DataFrame(to_predict_dict,index=[0]) 
    prediction, prediction_prob = predict_data(to_predict_df,model)
    return render_template('predict.html', prediction = prediction,  coupon = to_predict_df['coupon'][0] )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

    
    
    
    
