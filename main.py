#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def main():
    from sklearn.neighbors import KNeighborsClassifier
    pd.set_option('display.max_columns', None)
    CSV_FILE_PATH = 'kc_house_data_v2.csv'
    data = pd.read_csv(CSV_FILE_PATH) #https://www.jianshu.com/p/7ac36fafebea

    # delet data without output
    data['price'] = data['price'].fillna(1)
    data = data.drop(data[(data.price == 1)].index.tolist())
    
    target = data.loc[:, ['price']]

    # data=>train+validation+test 8(8:2):2
    x, Te_x, y, Te_y = train_test_split(data, target, test_size=0.2, train_size=0.8, random_state=1)
    Tr_x, Va_x, Tr_y, Va_y = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=50)
    
    x = x.fillna(x.mean())
    Te_x = Te_x.fillna(x.mean())
    Numerical_x = x.loc[:, ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                  'waterfront', 'views', 'condition', 'grade', 'sqft_above',
                                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                  'sqft_living15', 'sqft_lot15']]
    Te_Numerical_x = Te_x.loc[:, ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                  'waterfront', 'views', 'condition', 'grade', 'sqft_above',
                                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                  'sqft_living15', 'sqft_lot15']]
    
    scaler = preprocessing.StandardScaler().fit(Numerical_x)
    Te_x = scaler.transform(Te_Numerical_x)
    Te_xs = pd.DataFrame(data=Te_x, columns=Numerical_x.columns)
    x = scaler.transform(Numerical_x)
    xs = pd.DataFrame(data=x, columns=Numerical_x.columns)
    CM = xs.corr()
    cols = abs(CM).nlargest(21, 'price')['price'].index
    Te_x = Te_xs[cols.drop(['price']).drop(['yr_built']).drop(['long']).drop(['condition']).drop(['zipcode'])]
    x = xs[cols.drop(['price']).drop(['yr_built']).drop(['long']).drop(['condition']).drop(['zipcode'])]
    
    Te_y_average = np.mean(Te_y)
    array_mean = Te_y_average['price'] * np.ones([len(Te_y), 1])
    
    print('The baseline for the test data:')
    print('MSE for baseline= ', mean_squared_error(Te_y, array_mean))
    print('Error Rate for baseline=', 1 - r2_score(Te_y, array_mean))
    
    Err_out_of_sample = []
    mse_Te = []
    Err_Te = []
    mse_x = []
    Err_x = []
    
    for k in range(10):
        train1, pre_Tr_x_pick, train2, pre_Tr_y_pick = train_test_split(x, y, test_size=1 / 3)
        RF = RandomForestRegressor(n_estimators=47, bootstrap=True, random_state=0, oob_score=True)
        RF.fit(pre_Tr_x_pick, pre_Tr_y_pick.values.ravel())
        Err_x = np.append(Err_x, 1 - RF.score(x, y))
        Err_Te = np.append(Err_Te, 1 - RF.score(Te_x, Te_y))
        Err_out_of_sample = np.append(Err_out_of_sample, 1 - RF.oob_score_)
        mse_x = np.append(mse_x, mean_squared_error(RF.predict(x), y))
        mse_Te = np.append(mse_Te, mean_squared_error(RF.predict(Te_x), Te_y))
    print('For Random Forest Regression(47 trees):')
    print('In old training set:')
    print('Error Rate(mean) =', np.mean(Err_x))
    print('MSE(mean) =', np.mean(mse_x))
    
    print('In the test set:')
    print('Error Rate(mean) = %.5f and the variance = %.5f ' % (np.mean(Err_Te), np.var(Err_Te)))
    print('MSE(mean) = %.3f and the variance = %.3f ' % (np.mean(mse_Te), np.var(mse_Te)))
    print('Out Of Sample Error = ', np.mean(Err_out_of_sample))

if __name__ == "__main__":
    main()


# In[ ]:




