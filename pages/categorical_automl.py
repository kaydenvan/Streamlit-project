# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:21:34 2021

@author: van_s
"""
import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report\
    # , plot_confusion_matrix
from func.download_file import download_file
from func.upload_file import upload_file
import warnings
warnings.filterwarnings('ignore')

def iris_dataset():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    return df

@st.cache
def change_col_name(df):
    """
    Purpose: this function aims to convert the columns name into small letters.

    """
    cols_name = [x.lower().strip() for x in df.columns]
    df.columns = cols_name
    return df

def automl(algo, x_train, x_test, y_train, y_test):
    """ 
    Purpose: Taking the classification algorithm to create model and get performance report.
    """
    model = algo.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_acc = model.score(x_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_true=y_test, y_pred=y_pred, average='micro')
    test_rec = recall_score(y_true=y_test, y_pred=y_pred, average='micro')
    cross_val = cross_val_score(model, x_test, y_test, cv=3).mean()
    
    st.markdown(f'**Algorithm name: {model.__class__.__name__}**')
    st.markdown(f'*Training accuracy*: {train_acc:.4f}, *Testing accuracy*: {test_acc:.4f}, *Testing precision*: {test_prec:.4f}, *Testing recall*: {test_rec:.4f}')
    st.write(f'Cross validation score: {cross_val:.4f}')
    st.text('Model Report:\n' + classification_report(y_true=y_test, y_pred=y_pred))
    return model

def download_pred_df(df, features, target, model):
    """
    Purpose: create a link to download result file
    """
    df['y_pred'] = model.predict(df[features])
    df['accr'] = np.where(df[target] == df['y_pred'], True, False)
    df = df[[target] + ['y_pred', 'accr'] + features]
    st.markdown('download function is currently disabled')
    # download(df)

def categorical_automl():
    # main
    st.title('Auto Categorical ML')
    st.write('This app is powered by Streamlit, Sklearn, XGBoost, CatBoost and LightGBM')
    df, uploaded = upload_file(file_type = ['csv', 'xlsx', 'xls'], show_file_info = True)
    if not uploaded:
        demo = st.sidebar.radio('Enable Demo', ('Yes', 'No'), index=1,
                                help='Iris dataset is used for demonstration purpose')
        if demo == 'Yes':
            df = iris_dataset()
    else:
        demo = 'No'
    
    if df.empty:
        st.stop()    
    
    st.subheader('Preview uploaded dataframe') if uploaded else st.subheader('Preview demo dataframe')
    st.dataframe(df.head())
    st.markdown('*If you would like to do EDA for the dataset, please reach to the EDA page accordingly*')
    
    training = st.sidebar.number_input('Training ratio:', min_value=.1, max_value=1., 
                               value=.7, key='training_ratio', 
                               help='At least 10% of the data has to be trained') if demo == 'No' else .7
    st.write(f'The model will use {training*100:.2f}% data as the training set and the remaining as testing set')
    
    target = st.sidebar.selectbox('Please input target variable:', options=df.columns, key='target')\
        if demo == 'No' else 'target'
        
    if target != '' and target not in df.columns:
        st.error("target is not found in the dataset")
        st.stop()
    elif target == '':
        st.stop()
        
    # potential feature columns
    options = list(df.columns)
    options.remove(target)
    
    features = st.sidebar.multiselect('How many features do you want to keep?',
                             options,
                             default=options,
                             help='By default, the program will take all the columns as features. It is suggested to remove identifier since it should be useless.')\
        if demo == 'No' else options
    
    feature_processing = st.sidebar.checkbox('Please confirm the features before further processing')\
        if demo == 'No' else True
    if not feature_processing:
        st.stop()
    
    # transform target variable
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    
    with st.spinner('Quick Data Transformation and Cleansing are in process'):
        imp_median = SimpleImputer(strategy='median')
        imp_freq = SimpleImputer(strategy='most_frequent')
        minmax = MinMaxScaler()
        oe = OrdinalEncoder()
        
        for col in df[features].select_dtypes(include=['number']):
            df[col] = imp_median.fit_transform(np.array(df[col]).reshape(-1, 1))
            df[col] = minmax.fit_transform(np.array(df[col]).reshape(-1, 1))
        for col in df[features].select_dtypes(include=['object']):
            df[col] = imp_freq.fit_transform(np.array(df[col]).reshape(-1, 1))
            df[col] = oe.fit_transform(np.array(df[col]).reshape(-1, 1))
    
    st.write('For the purpose of data modeling, the dataset will be transformed accordingly')
    st.write('Preview transformed dataset')
    st.dataframe(df[[target] +features].head())
    
    options = st.sidebar.multiselect('Which models do you want to create?', 
                             ['Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Logistic Regression'],
                             default=['XGBoost'],
                             help='By default, the program will develop a XGBoost model as it is in general with high accuracy. You can choose more for comparison.')\
        if demo == 'No' else ['Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Logistic Regression']
    
    model_processing = st.sidebar.button('Confirm')\
        if demo == 'No' else True
    if not model_processing:
        st.stop()  
    if len(options) <= 0:
        st.error('No model is selected')
        st.stop()
    
    st.title('Data Modeling')
    
    # split train test set
    with st.spinner(f'Training {round(df.shape[0]*training)} data'):
        x_train, x_test, y_train, y_test = train_test_split(df[features], 
                                                            df[target], 
                                                            train_size=training, 
                                                            random_state=1)
    
    if 'Random Forest' in options:
        with st.spinner('Model development in progress'):
            tree = RandomForestClassifier(random_state=1)
            model = automl(tree, x_train, x_test, y_train, y_test)
            download_pred_df(df, features, target, model)
    if 'XGBoost' in options:
        with st.spinner('Model development in progress'):
            xgb = XGBClassifier()
            model = automl(xgb, x_train, x_test, y_train, y_test)
            download_pred_df(df, features, target, model)
    if 'CatBoost' in options:
        with st.spinner('Model development in progress'):
            cat = CatBoostClassifier(random_seed=1, verbose=0)
            model = automl(cat, x_train, x_test, y_train, y_test)
            download_pred_df(df, features, target, model)
    if 'LightGBM' in options:
        with st.spinner('Model development in progress'):
            lgb = LGBMClassifier()
            model = automl(lgb, x_train, x_test, y_train, y_test)
            download_pred_df(df, features, target, model)
    if 'Logistic Regression' in options:
        with st.spinner('Model development in progress'):
            logit = LogisticRegression(random_state=1)
            model = automl(logit, x_train, x_test, y_train, y_test)
            download_pred_df(df, features, target, model)


# fig, ax = plt.subplots(figsize=(4,4))
# plot_confusion_matrix(model, x_test, y_test, display_labels=model.classes_,
#                      normalize='true', cmap=plt.cm.Blues, ax=ax)
# plt.title('Normalized Confusion Matrix')
# st.pyplot(fig)