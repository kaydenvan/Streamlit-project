# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 03:38:07 2021

@author: van_s
"""

import pandas as pd
import streamlit as st
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from func.upload_file import upload_file
from func.download_file import download_file

warnings.filterwarnings('ignore')

def auto_xgboost_regression():
    # for regression problem
    st.title('Auto Regression')
    st.write('This app is powered by Streamlit, Sklearn, XGBoost.')
    st.write("""This page aims to facilitate the Kaggle competition submission.
             You have to update train and test dataset. This page aims to 
             demonstrate regression model and with pipeline developed""")
             
    col_train, col_test = st.beta_columns(2)
    with col_train:
        st.write('Upload training dataset')
        train = pd.DataFrame() 
        train, train_uploaded, file_name = upload_file(file_type = ['csv', 'xlsx', 'xls'], 
                                                       return_file_name = True,
                                                       key='upload_train')
    
    with col_test:
        st.write('Upload testing dataset')
        test = pd.DataFrame()
        test, test_uploaded = upload_file(file_type = ['csv', 'xlsx', 'xls'], 
                                                       key='upload_test')
        
    if not train_uploaded:
        st.stop()
        
    show_upload = st.beta_expander('Preview train dataframe', expanded=True) if train_uploaded\
             else st.beta_expander('Preview demo dataframe', expanded=True)
    show_upload.dataframe(train.head(50))
    
    # data cleansing
    st.write("""For columns that over 50% of them are null, 
             it will be removed.""")
    null_cols = train.isnull().sum()/train.shape[0]
    null_cols = null_cols[null_cols>50].keys()
    train.drop(columns=null_cols, inplace=True)
    
    y_ = st.sidebar.selectbox('Select target variable', train.select_dtypes(include='number').columns.to_list(),
                              help='Please choose the target label')
    identifier_ = st.sidebar.selectbox('Select identifier (Optional)', [None]+train.columns.to_list())
    
    x_ = train.columns.to_list()
    x_.remove(y_)
    
    # validate if test dataset contains required information
    exist_ = test.columns.isin(x_).all()
    
    if exist_:
        if y_ in test.columns:
            x_train, x_test, y_train, y_test = train[x_], test[x_], train[y_], test[y_]
        else:
            x_train, x_test, y_train = train[x_], test[x_], train[y_]
    else:
        x_train, x_test, y_train, y_test = train_test_split(train[x_], train[y_], train_size=.7)
    
    # use pipeline to train
    # find object cols
    cat_cols = train[x_].select_dtypes(include='object').columns.to_list()
    
    # find numeric cols
    num_cols = train[x_].select_dtypes(include='number').columns.to_list()
    
    # prepare Pipeline
    categorical_transformation = Pipeline(steps=[
        ('most frequent impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])
    
    numerical_transformation = Pipeline(steps=[
        ('median impute', SimpleImputer(strategy='median')),
        ('standardization', StandardScaler()),
        ])
    
    preprocessing = ColumnTransformer(
        transformers=[
            ('numeric', numerical_transformation, num_cols),
            ('categoric', categorical_transformation, cat_cols),
            ]
        )
    
    xgb = XGBRegressor(n_estimator=100, max_depth=8, silent=True, random_seed=0)
    
    # create pipeline
    clf = Pipeline(steps=[
        ('preprocess', preprocessing),
        ('xgb model', xgb),
        ])
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accr = clf.score(train[x_], train[y_])
    st.write(f'Training Accuracy: {accr*100:.4f}')
    
    if not test_uploaded:
        if identifier_ != None:
            result = pd.DataFrame({identifier_: train[identifier_], 'y':train[y_], 'y_pred':clf.predict(train[x_])})
        else:
            result = pd.DataFrame({'y':train[y_], 'y_pred':clf.predict(train[x_])})
    else:
        if y_ in test.columns:
            if identifier_ != None:
                result = pd.DataFrame({identifier_: test[identifier_], 'y':test[y_], 'y_pred':clf.predict(test[x_])})
            else:
                result = pd.DataFrame({'y':test[y_], 'y_pred':clf.predict(test[x_])})
        else:
            if identifier_ != None:
                result = pd.DataFrame({identifier_: test[identifier_], 'y_pred':clf.predict(test[x_])})
            else:
                result = pd.DataFrame({'y_pred':clf.predict(test[x_])})
    
    if 'y' in result.columns:
        result['variance'] = (result['y_pred'] - result['y'])/result['y']*100
        
    st.write('result dataframe')
    st.dataframe(result.head(50))
    download_file(result)