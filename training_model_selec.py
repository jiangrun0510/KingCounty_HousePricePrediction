#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge



def without_outlier(Tr_x, Q1, Q3, IQR):
    ColumnName = Tr_x.columns
    outlier_detect_column = ColumnName[3:len(ColumnName) - 1]
    Tr_x_out = Tr_x[~((Tr_x[outlier_detect_column] < (Q1[outlier_detect_column] - 1.5 * IQR[outlier_detect_column])) | (
                    Tr_x[outlier_detect_column] > (Q3[outlier_detect_column] + 1.5 * IQR[outlier_detect_column]))).any(axis=1)]

    Tr_x.loc[Tr_x.bedrooms > (Q3.bedrooms + 1.5 * IQR.bedrooms), 'bedrooms'] = max(Tr_x_out.bedrooms)
    Tr_x.loc[Tr_x.bedrooms < (Q1.bedrooms - 1.5 * IQR.bedrooms), 'bedrooms'] = min(Tr_x_out.bedrooms)

    Tr_x.loc[Tr_x.bathrooms > (Q3.bathrooms + 1.5 * IQR.bathrooms), 'bathrooms'] = max(Tr_x_out.bathrooms)
    Tr_x.loc[Tr_x.bathrooms < (Q1.bathrooms - 1.5 * IQR.bathrooms), 'bathrooms'] = min(Tr_x_out.bathrooms)

    Tr_x.loc[Tr_x.sqft_living > (Q3.sqft_living + 1.5 * IQR.sqft_living), 'sqft_living'] = max(Tr_x_out.sqft_living)
    Tr_x.loc[Tr_x.sqft_living < (Q1.sqft_living - 1.5 * IQR.sqft_living), 'sqft_living'] = min(Tr_x_out.sqft_living)

    Tr_x.loc[Tr_x.sqft_lot > (Q3.sqft_lot + 1.5 * IQR.sqft_lot), 'sqft_lot'] = max(Tr_x_out.sqft_lot)
    Tr_x.loc[Tr_x.sqft_lot < (Q1.sqft_lot - 1.5 * IQR.sqft_lot), 'sqft_lot'] = min(Tr_x_out.sqft_lot)

    Tr_x.loc[Tr_x.floors > (Q3.floors + 1.5 * IQR.floors), 'floors'] = max(Tr_x_out.floors)
    Tr_x.loc[Tr_x.floors < (Q1.floors - 1.5 * IQR.floors), 'floors'] = min(Tr_x_out.floors)

    Tr_x.loc[Tr_x.waterfront > (Q3.waterfront + 1.5 * IQR.waterfront), 'waterfront'] = max(Tr_x_out.waterfront)
    Tr_x.loc[Tr_x.waterfront < (Q1.waterfront - 1.5 * IQR.waterfront), 'waterfront'] = min(Tr_x_out.waterfront)

    Tr_x.loc[Tr_x.views > (Q3.views + 1.5 * IQR.views), 'views'] = max(Tr_x_out.views)
    Tr_x.loc[Tr_x.views < (Q1.views - 1.5 * IQR.views), 'views'] = min(Tr_x_out.views)

    Tr_x.loc[Tr_x['condition'] > (Q3['condition'] + 1.5 * IQR['condition']), 'condition'] = max(Tr_x_out['condition'])
    Tr_x.loc[Tr_x['condition'] < (Q1['condition'] - 1.5 * IQR['condition']), 'condition'] = min(Tr_x_out['condition'])

    Tr_x.loc[Tr_x.grade > (Q3.grade + 1.5 * IQR.grade), 'grade'] = max(Tr_x_out.grade)
    Tr_x.loc[Tr_x.grade < (Q1.grade - 1.5 * IQR.grade), 'grade'] = min(Tr_x_out.grade)

    Tr_x.loc[Tr_x.sqft_above > (Q3.sqft_above + 1.5 * IQR.sqft_above), 'sqft_above'] = max(Tr_x_out.sqft_above)
    Tr_x.loc[Tr_x.sqft_above < (Q1.sqft_above - 1.5 * IQR.sqft_above), 'sqft_above'] = min(Tr_x_out.sqft_above)

    Tr_x.loc[Tr_x.sqft_basement > (Q3.sqft_basement + 1.5 * IQR.sqft_basement), 'sqft_basement'] = max(Tr_x_out.sqft_basement)
    Tr_x.loc[Tr_x.sqft_basement < (Q1.sqft_basement - 1.5 * IQR.sqft_basement), 'sqft_basement'] = min(Tr_x_out.sqft_basement)

    Tr_x.loc[Tr_x.yr_built > (Q3.yr_built + 1.5 * IQR.yr_built), 'yr_built'] = max(Tr_x_out.yr_built)
    Tr_x.loc[Tr_x.yr_built < (Q1.yr_built - 1.5 * IQR.yr_built), 'yr_built'] = min(Tr_x_out.yr_built)

    Tr_x.loc[Tr_x.yr_renovated > (Q3.yr_renovated + 1.5 * IQR.yr_renovated), 'yr_renovated'] = max(Tr_x_out.yr_renovated)
    Tr_x.loc[Tr_x.yr_renovated < (Q1.yr_renovated - 1.5 * IQR.yr_renovated), 'yr_renovated'] = min(Tr_x_out.yr_renovated)

    Tr_x.loc[Tr_x.zipcode > (Q3.zipcode + 1.5 * IQR.zipcode), 'zipcode'] = max(Tr_x_out.zipcode)
    Tr_x.loc[Tr_x.zipcode < (Q1.zipcode - 1.5 * IQR.zipcode), 'zipcode'] = min(Tr_x_out.zipcode)

    Tr_x.loc[Tr_x['lat'] > (Q3['lat'] + 1.5 * IQR['lat']), 'lat'] = max(Tr_x_out['lat'])
    Tr_x.loc[Tr_x['lat'] < (Q1['lat'] - 1.5 * IQR['lat']), 'lat'] = min(Tr_x_out['lat'])

    Tr_x.loc[Tr_x['long'] > (Q3['long'] + 1.5 * IQR['long']), 'long'] = max(Tr_x_out['long'])
    Tr_x.loc[Tr_x['long'] < (Q1['long'] - 1.5 * IQR['long']), 'long'] = min(Tr_x_out['long'])

    Tr_x.loc[Tr_x.sqft_living15 > (Q3.sqft_living15 + 1.5 * IQR.sqft_living15), 'sqft_living15'] = max(Tr_x_out.sqft_living15)
    Tr_x.loc[Tr_x.sqft_living15 < (Q1.sqft_living15 - 1.5 * IQR.sqft_living15), 'sqft_living15'] = min(Tr_x_out.sqft_living15)

    Tr_x.loc[Tr_x.sqft_lot15 > (Q3.sqft_lot15 + 1.5 * IQR.sqft_lot15), 'sqft_lot15'] = max(Tr_x_out.sqft_lot15)
    Tr_x.loc[Tr_x.sqft_lot15 < (Q1.sqft_lot15 - 1.5 * IQR.sqft_lot15), 'sqft_lot15'] = min(Tr_x_out.sqft_lot15)

    return Tr_x


def main():
    from sklearn.neighbors import KNeighborsClassifier
    pd.set_option('display.max_columns', None)
    CSV_FILE_PATH = 'kc_house_data_v2.csv'
    data = pd.read_csv(CSV_FILE_PATH) #https://www.jianshu.com/p/7ac36fafebea

    # delet data without output
    data['price'] = data['price'].fillna(1)
    data = data.drop(data[(data.price == 1)].index.tolist())

    labels = data.loc[:, ['price']]
    
    
    # data=>train+validation+test 8(8:2):2
    x, Te_x, y, Te_y = train_test_split(data, labels, test_size=0.2, train_size=0.8, random_state=1)
    Tr_x, Va_x, Tr_y, Va_y = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=50)
    
    Null_train_ratio = ((Tr_x.isnull().sum() / len(Tr_x)) * 100).sort_values(ascending=False)
    missing_train_ratio = pd.DataFrame({'Missing train data ratio': Null_train_ratio.drop(Null_train_ratio[Null_train_ratio == 0].index)})

    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')  # feature rotation
    sns.barplot(x=Null_train_ratio.index, y=Null_train_ratio)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('missing data percentage by feature', fontsize=15)
    plt.show()


    # compare "with outliers" and "without outliers"
    Q1 = Tr_x.quantile(0.25)
    Q3 = Tr_x.quantile(0.75)
    IQR = Q3 - Q1
    #Tr_x = without_outlier(Tr_x, Q1, Q3, IQR)
    #Va_x = without_outlier(Va_x, Q1, Q3, IQR)
    
    #compare mean, median and 0
    #using mean value
    Tr_x = Tr_x.fillna(Tr_x.mean())
    Va_x = Va_x.fillna(Tr_x.mean())
    x = x.fillna(x.mean())
    Te_x = Te_x.fillna(x.mean())
    
    #using median
    #Tr_x=Tr_x.fillna(Tr_x.median())
    #Va_x=Va_x.fillna(Tr_x.median())
    #x = x.fillna(x.median())
    #Te_x = Te_x.fillna(x.median())
    
    #using 0
    #Tr_x=Tr_x.fillna(0)
    #Va_x=Va_x.fillna(0)
    #x = x.fillna(0)
    #Te_x = Te_x.fillna(0)

    
    
    # importing one hot encoder from sklearn 
    # There are changes in OneHotEncoder class 
    #from sklearn.preprocessing import OneHotEncoder 
    #from sklearn.compose import ColumnTransformer 
   
    # creating one hot encoder object with categorical feature 0 
    # indicating the first column 
    #columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') 
    #data = np.array(columnTransformer.fit_transform(data), dtype = np.str) 
    #drop_enc = OneHotEncoder(drop='first').fit(X)
    #drop_enc.categories_[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    #drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()array([[0., 0., 0.],[1., 1., 0.]])
    
    
    # separate the categoty feature and numerical features
    Tr_x_str = Tr_x.loc[:, ["date"]]
    Tr_x_num = Tr_x.loc[:, ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                  'waterfront', 'views', 'condition', 'grade', 'sqft_above',
                                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                  'sqft_living15', 'sqft_lot15']]
    Va_x_str = Va_x.loc[:, ["date"]]
    Va_x_num = Va_x.loc[:, ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                  'waterfront', 'views', 'condition', 'grade', 'sqft_above',
                                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                  'sqft_living15', 'sqft_lot15']]
    x_str = x.loc[:, ["date"]]
    x_num = x.loc[:, ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                  'waterfront', 'views', 'condition', 'grade', 'sqft_above',
                                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                  'sqft_living15', 'sqft_lot15']]
    Te_x_num = Te_x.loc[:, ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                  'waterfront', 'views', 'condition', 'grade', 'sqft_above',
                                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                                  'sqft_living15', 'sqft_lot15']]

    Tr_x_str = pd.get_dummies(Tr_x_str)
    date_col = Tr_x_str.columns
    
    Tr_x_str["price"] = Tr_y

    date_CM = Tr_x_str.corr()
    cols = abs(date_CM).nlargest(8, 'price')['price'].index
    cm = np.corrcoef(Tr_x_str[cols].values.T)
    sns.set(font_scale=1.25)
    plt.subplots(figsize=(15, 12))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('The date that is most related to price')
    plt.show()
    
    # standardization
    standar = preprocessing.StandardScaler().fit(Tr_x_num)
    Tr_x = standar.transform(Tr_x_num)
    Va_x = standar.transform(Va_x_num)
    Tr_x1 = pd.DataFrame(data=Tr_x, columns=Tr_x_num.columns)
    Va_x1 = pd.DataFrame(data=Va_x, columns=Tr_x_num.columns)
    
    # Correlation with house price
    CM = Tr_x1.corr()
    plt.subplots(figsize=(18, 15))
    ax = sns.heatmap(CM, vmax=1, annot=True, square=True, vmin=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    cols = abs(CM).nlargest(19, 'price')['price'].index
    cm = np.corrcoef(Tr_x1[cols].values.T)
    sns.set(font_scale=1.25)
    plt.subplots(figsize=(15, 12))
    plt.title('18 Features most related to price')
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    
    cols = abs(CM).nlargest(21, 'price')['price'].index
    
    column_R = cols.drop(['price']).drop(['yr_built']).drop(['long']).drop(['condition']).drop(['zipcode'])
    
    # test set transformation
    scaler = preprocessing.StandardScaler().fit(x_num)
    x = scaler.transform(x_num)
    Te_x = scaler.transform(Te_x_num)
    x1 = pd.DataFrame(data=x, columns=Tr_x_num.columns)
    Te_x1 = pd.DataFrame(data=Te_x, columns=Tr_x_num.columns)

    Tr_x = Tr_x1[column_R]
    Va_x = Va_x1[column_R]
    x = x1[column_R]
    Te_x = Te_x1[column_R]
    
    
    # using Random Forest Regression to see the feature importance
    #RF regression plot
    Err_tra_mean = []
    Err_val_mean = []
    MSE_tra_mean = []
    MSE_val_mean = []
    for i in range(50):
        error_train = []
        error_val = []
        mse_train = []
        mse_val = []
        for j in range(10):
            pre_train, pre_Tr_x_pick, pre_Tr_y, pre_Tr_y_pick = train_test_split(Tr_x, Tr_y,test_size=1 / 3)
            RF = RandomForestRegressor(n_estimators=i + 1, bootstrap=True, random_state=0)
            RF.fit(pre_Tr_x_pick, pre_Tr_y_pick.values.ravel())
            error_train = np.append(error_train, 1 - RF.score(Tr_x, Tr_y))
            mse_train = np.append(mse_train, mean_squared_error(RF.predict(Tr_x), Tr_y))
            error_val = np.append(error_val, 1 - RF.score(Va_x, Va_y))
            mse_val = np.append(mse_val, mean_squared_error(RF.predict(Va_x), Va_y))
        Err_tra_mean = np.append(Err_tra_mean, np.mean(error_train))
        Err_val_mean = np.append(Err_val_mean, np.mean(error_val))
        MSE_tra_mean = np.append(MSE_tra_mean, np.mean(mse_train))
        MSE_val_mean = np.append(MSE_val_mean, np.mean(mse_val))
    print('Random Forest Regression:')
    print("In validation set, minimum MSE =%.3f, error rate=%.3f for %.0f trees" % (min(MSE_val_mean), min(Err_val_mean), np.argmin(MSE_val_mean) + 1))
    fea_importance = pd.DataFrame({'feature': list(Tr_x.columns),
                                   'importance': RF.feature_importances_}).\
                    sort_values('importance', ascending=False)
    print('importance=', fea_importance)

    R = np.arange(1, 51)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('Random Forest Regression')
    ax1.plot(R, Err_tra_mean, label='data in training set')
    ax1.plot(R, Err_val_mean, color='r', label='data in validation set')
    ax1.set_ylabel('Error Rate')
    ax1.plot(1+np.argmin(Err_val_mean), min(Err_val_mean), '*', label='minimum', color='b', markersize=15)
    ax1.legend(loc='best')

    ax2.plot(R, MSE_tra_mean, label='data in training set')
    ax2.plot(R, MSE_val_mean, color='r', label='data in validation set')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Number of trees')
    ax2.plot(1+np.argmin(MSE_val_mean), min(MSE_val_mean), '*', label='minimum', color='b', markersize=15)
    ax2.legend(loc='best')
    plt.show()
    
    
    #linear regression
    
    Err_val_linear = []
    Err_tra_linear = []
    MSE_val_linear = []
    MSE_tra_linear = []
    CM = Tr_x1.corr()
    for i in range(5, 21):
        cols = abs(CM).nlargest(i, 'price')['price'].index
        column_R = cols.drop(['price'])
        Tr_x = Tr_x1[column_R]
        Va_x = Va_x1[column_R]
        reg = linear_model.LinearRegression()
        reg.fit(Tr_x, Tr_y)
        MSE_val_linear = np.append(MSE_val_linear, mean_squared_error(Va_y, reg.predict(Va_x)))
        MSE_tra_linear = np.append(MSE_tra_linear, mean_squared_error(Tr_y, reg.predict(Tr_x)))
        Err_val_linear = np.append(Err_val_linear, 1 - reg.score(Va_x, Va_y))
        Err_tra_linear = np.append(Err_tra_linear, 1 - reg.score(Tr_x, Tr_y))
    print('Linear regression:')
    print('minimum error rate= %.4f , minimum MSE= %.4f while using %.0f features' % 
          (min(Err_val_linear), min(MSE_val_linear), 4+np.argmin(MSE_val_linear)))

    Rg = np.arange(4, 20, 1)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('Linear Regression')
    ax1.plot(Rg, Err_val_linear, color='r', label='Validation set')
    ax1.plot(Rg, Err_tra_linear, label='Training set')
    ax1.set_ylabel('Error Rate')
    ax1.plot(4+np.argmin(Err_val_linear), min(Err_val_linear), '*', label='minimum', color='b', markersize=15)
    ax1.legend(loc='best')
    
    ax2.plot(Rg, MSE_val_linear, color='r', label='Validation set')
    ax2.plot(Rg, MSE_tra_linear, label='Training set')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Number of Features')
    ax2.plot(4+np.argmin(MSE_val_linear), min(MSE_val_linear), '*', label='minimum', color='b', markersize=15)
    ax2.legend(loc='best')
    plt.show()
    
    
    # Ridge Regression(without CV)
    #from sklearn.linear_model import Ridge
    #import numpy as np
    #n_samples, n_features = 10, 5
    #rng = np.random.RandomState(0)
    #y = rng.randn(n_samples)
    #X = rng.randn(n_samples, n_features)
    #clf = Ridge(alpha=1.0)
    #clf.fit(X, y)
    #Ridge()
    
    R = np.linspace(-4, 1, 50)
    cols = abs(CM).nlargest(19, 'price')['price'].index
    column_R = cols.drop(['price'])
    Tr_x = Tr_x1[column_R]
    Va_x = Va_x1[column_R]
    MSE_val_ridge = []
    MSE_tra_ridge = []
    Err_val_ridge = []
    Err_tra_ridge = []
    for i in R:
        ridge = linear_model.Ridge(alpha=10 ** i, normalize=True)
        ridge.fit(Tr_x, Tr_y)
        MSE_val_ridge = np.append(MSE_val_ridge, mean_squared_error(Va_y, ridge.predict(Va_x)))
        MSE_tra_ridge = np.append(MSE_tra_ridge, mean_squared_error(Tr_y, ridge.predict(Tr_x)))
        Err_val_ridge = np.append(Err_val_ridge, 1 - ridge.score(Va_x, Va_y))
        Err_tra_ridge = np.append(Err_tra_ridge, 1 - ridge.score(Tr_x, Tr_y))
    print('Ridge Regression(without CV):')
    print('We can find the min error rate= %.4f and min MSE= %.4f when alpha= %.6f ' % (
    min(Err_val_ridge), min(MSE_val_ridge), 10 ** R[np.argmin(MSE_val_ridge)]))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('Ridge Regression (Without CV)')
    ax1.plot(R, Err_tra_ridge, label='Training set')
    ax1.plot(R, Err_val_ridge, color='r', label='Validation set')
    ax1.set_ylabel('Error Rate')
    ax1.set_xlabel('log(alpha)')
    ax1.plot(R[np.argmin(Err_val_ridge)], min(Err_val_ridge), '*', label='minimum', color='b', markersize=15)
    ax1.legend(loc='best')

    ax2.plot(R, MSE_tra_ridge, label='Training set')
    ax2.plot(R, MSE_val_ridge, color='r', label='Validation set')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('log(alpha)')
    ax2.plot(R[np.argmin(MSE_val_ridge)], min(MSE_val_ridge), '*', label='minimum', color='b', markersize=15)
    ax2.legend(loc='best')
    plt.show()
    
    
    
    
    #Ridge Regression(with CV)
    R = np.linspace(-4, 1, 50)
    cols = abs(CM).nlargest(19, 'price')['price'].index
    column_R = cols.drop(['price']).drop(['long'])

    xx = x1[column_R]
    Err_CV_ridge = np.zeros([len(R), 6])
    MSE_CV_ridge = np.zeros([len(R), 6])
    for i in range(5, 11):  # column
        kfold = KFold(n_splits=i, shuffle=True)
        for j in range(len(R)):  # row
            ridgeCV = linear_model.RidgeCV(alphas=10 ** R, normalize=True)
            Err_CV_ridge[j][i-5] = 1- np.mean(cross_val_score(ridgeCV, xx, y, cv=kfold, scoring='r2'))
            MSE_CV_ridge[j][i-5] = np.mean(cross_val_score(ridgeCV, xx, y, cv=kfold, scoring='neg_mean_squared_error'))*(-1)
    Idx_minErr = np.unravel_index(Err_CV_ridge.argmin(), Err_CV_ridge.shape)
    Idx_minMSE = np.unravel_index(MSE_CV_ridge.argmin(), MSE_CV_ridge.shape)
    print('Ridge Regression(with CV):')
    print('We got the min MSE value= %.3f when we applied %.0f fold and alpha = %.5f' % (
    MSE_CV_ridge.min(), Idx_minMSE[1] + 5, 10 ** R[Idx_minMSE[0]]))
    print('We can find the min error rate= %.4f' % (Err_CV_ridge.min()))
    
    K_a_MSEpair = MSE_CV_ridge[:, Idx_minMSE[1]].reshape((MSE_CV_ridge[:, Idx_minMSE[1]].shape[0], 1))
    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('Ridge Regression (With CV when K= %.0f)' % (5+Idx_minMSE[1]))

    ax2.plot(R, K_a_MSEpair)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('log(alpha)')
    ax2.plot(R[Idx_minMSE[0]], K_a_MSEpair.min(), '*', label='minimum', color='b', markersize=15)
    ax2.legend(loc='best')
    plt.show()
    
    
    #Lasso Regression(without CV)
    #clf = linear_model.Lasso(alpha=0.1)
    #clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    #Lasso(alpha=0.1)
    #print(clf.coef_)
    #print(clf.intercept_)
    

    R = np.linspace(-4, 1, 50)
    Tr_x = Tr_x1[column_R]
    Va_x = Va_x1[column_R]
    MSE_val_lasso = []
    MSE_tra_lasso = []
    Err_val_lasso = []
    Err_tra_lasso = []
    for i in R:
        lasso = linear_model.Lasso(alpha=10 ** i, normalize=True)
        lasso.fit(Tr_x, Tr_y)
        MSE_val_lasso = np.append(MSE_val_lasso, mean_squared_error(Va_y, lasso.predict(Va_x)))
        MSE_tra_lasso = np.append(MSE_tra_lasso, mean_squared_error(Tr_y, lasso.predict(Tr_x)))
        Err_val_lasso = np.append(Err_val_lasso, 1 - lasso.score(Va_x, Va_y))
        Err_tra_lasso = np.append(Err_tra_lasso, 1 - lasso.score(Tr_x, Tr_y))
    print('Lasso Regression(without CV):')
    print('We can find the min error rate= %.4f and min MSE= %.4f when alpha= %.6f ' % (
    min(Err_val_lasso), min(MSE_val_lasso), 10 ** R[np.argmin(MSE_val_lasso)]))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('Lasso Regression (Without CV)')
    ax1.plot(R, Err_tra_lasso, label='Training set')
    ax1.plot(R, Err_val_lasso, color='r', label='Validation set')
    ax1.set_ylabel('Error Rate')
    ax1.plot(R[np.argmin(Err_val_lasso)], min(Err_val_lasso), '*', label='minimum', color='b', markersize=15)
    ax1.legend(loc='lower right')

    ax2.plot(R, MSE_tra_lasso, label='Training set')
    ax2.plot(R, MSE_val_lasso, color='r', label='Validation set')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('log(alpha)')
    ax2.plot(R[np.argmin(MSE_val_lasso)], min(MSE_val_lasso), '*', label='minimum', color='b', markersize=15)
    plt.show()
    
    
    #Lasso Regression(with CV)
    R = np.linspace(-4, 1, 50)
    xx = x1[column_R]
    Err_CV_lasso = np.zeros([len(R), 6])
    MSE_CV_lasso = np.zeros([len(R), 6])
    for i in range(5, 11): 
        kfold = KFold(n_splits=i, shuffle=True)
        for j in range(len(R)):
            lassoCV = linear_model.LassoCV(alphas=10 ** R, normalize=True)
            Err_CV_lasso[j][i-5] = 1- np.mean(cross_val_score(lassoCV, xx, y, cv=kfold, scoring='r2'))
            MSE_CV_lasso[j][i-5] = np.mean(cross_val_score(lassoCV, xx, y, cv=kfold, scoring='neg_mean_squared_error'))*(-1)
    Idx_minErr=np.unravel_index(Err_CV_lasso.argmin(), Err_CV_lasso.shape)
    Idx_minMSE = np.unravel_index(MSE_CV_lasso.argmin(), MSE_CV_lasso.shape)
    print('Lasso Regression(with CV):')
    print('We got the min MSE value= %.3f when we applied %.0f fold and alpha = %.5f' % (
    MSE_CV_lasso.min(), Idx_minMSE[1] + 5, 10 ** R[Idx_minMSE[0]]))

    print('We can find the min error rate= %.4f' % (Err_CV_lasso.min()))
    
    
    K_a_MSEpair = MSE_CV_lasso[:, Idx_minMSE[1]].reshape((MSE_CV_lasso[:, Idx_minMSE[1]].shape[0], 1))
    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('Lasso Regression (With CV when K= %.0f)' % (Idx_minMSE[1] + 5))

    ax2.plot(R, K_a_MSEpair)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('log(alpha)')
    ax2.plot(R[Idx_minMSE[0]], K_a_MSEpair.min(), '*', label='minimum', color='b', markersize=15)
    ax2.legend(loc='best')

    plt.show()

    avg_ytest = np.mean(Te_y)
    one_array = np.ones([len(Te_y), 1])
    mean_arr = avg_ytest['price'] * one_array
    MSE_baseline = mean_squared_error(Te_y, mean_arr)
    Err_baseline = 1 - r2_score(Te_y, mean_arr)

    print('Number of training data (original):', len(y))
    print('Number of test data:', len(Te_y))
    print('Number of training data (New):', len(Tr_y))
    print('Number of validation data:', len(Va_y))

    
    
    
    #Random Forest Regression
    cols = abs(CM).nlargest(20, 'price')['price'].index
    column_R = cols.drop(['price']).drop(['yr_built']).drop(['long']).drop(['condition']).drop(['zipcode'])

    Err_x = []
    Err_te = []
    mse_x = []
    mse_te = []
    Err_out_of_sample = []
    for j in range(10):
        pre_train, pre_Tr_x_pick, pre_Tr_y, pre_Tr_y_pick = train_test_split(x, y, test_size=1 / 3)
        RF = RandomForestRegressor(n_estimators=47, bootstrap=True, random_state=0, oob_score=True)
        RF.fit(pre_Tr_x_pick, pre_Tr_y_pick.values.ravel())
        Err_x = np.append(Err_x, 1 - RF.score(x, y))
        Err_te = np.append(Err_te, 1 - RF.score(Te_x, Te_y))
        Err_out_of_sample = np.append(Err_out_of_sample, 1 - RF.oob_score_)
        mse_x = np.append(mse_x, mean_squared_error(RF.predict(x), y))
        mse_te = np.append(mse_te, mean_squared_error(RF.predict(Te_x), Te_y))
    mean_oob_err = np.mean(Err_out_of_sample)
    meanErr_x = np.mean(Err_x)
    meanErr_te = np.mean(Err_te)
    mean_mse_x = np.mean(mse_x)
    mean_mse_te = np.mean(mse_te)
    var_mse_te = np.var(mse_te)
    var_err_test = np.var(Err_te)
    
    print('Baseline')
    print('The baseline for the test data:')
    print('MSE = ', MSE_baseline)
    print('Error Rate=', Err_baseline)
    
    print('For Random Forest Regression(47 trees) model:')
    print('in the training set')
    print('MSE(mean) =', mean_mse_x)
    print('Error Rate(mean) =', meanErr_x)
    print('In the test set')
    print('MSE(mean) = %.3f and variance = %.3f ' % (mean_mse_te, var_mse_te))
    print('Error Rate(mean) = %.5f with variance = %.5f ' % (meanErr_te, var_err_test))
    print('Out Of Sample Error = ', mean_oob_err)
    
    

    #plots for the 4 most important featuees
    feature_4important = ['sqft_living', 'grade', 'sqft_living15', 'sqft_above']
    for i in feature_4important:
        for j in range(10):
            pre_train, pre_Tr_x_pick, pre_Tr_y, pre_Tr_y_pick = train_test_split(x, y, test_size=1 / 3)
            RF = RandomForestRegressor(n_estimators=47, bootstrap=True, random_state=0)
            RF.fit(pre_Tr_x_pick[[i]], pre_Tr_y_pick.values.ravel())
        Xrange = np.arange(min(x[i]), max(x[i]), 0.001)
        Xrange = Xrange.reshape((len(Xrange), 1))

        plt.scatter(x[i], y, color='blue', label='data in training set')
        plt.plot(Xrange, RF.predict(Xrange), color='green', label='regression function')
        plt.title('Random Forest Regression')
        plt.xlabel(i)
        plt.ylabel('price')
        plt.legend(loc='best')
        plt.show()
    
    
if __name__ == "__main__":
    main()
    


# In[ ]:




