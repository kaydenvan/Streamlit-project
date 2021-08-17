# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 13:36:24 2021

@author: van_s
"""
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, plot_tree, plot_importance
import xgboost as xgb
from func.download_file import download_file, download_model
from func.upload_file import upload_file
import warnings
warnings.filterwarnings('ignore')

@st.cache(allow_output_mutation=True)
def _get_best_iteration(x_train, x_test, y_train, y_test, params):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    params.update(dict(num_class =len(set(y_train))))
    model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, "Test")],
            early_stopping_rounds=10
            )
    params['num_boost_round'] = model.best_iteration+1
    return params

@st.cache(allow_output_mutation=True)
def _get_best_hyperparameters(x_train, y_train, params, **kwargs):
    # take configuration from input if any
    start_eta = kwargs.get('start_eta', .01)
    end_eta = kwargs.get('end_eta', .3)
    step_eta = (end_eta-start_eta)/5
    
    start_max_depth = kwargs.get('start_max_depth', 2)
    end_max_depth = kwargs.get('end_max_depth', 12)
    step_max_depth = math.ceil((end_max_depth-start_max_depth)/5)
    
    start_min_child_weight = kwargs.get('start_min_child_weight', 1)
    end_min_child_weight = kwargs.get('end_min_child_wegiht', 10)
    step_min_child_weight = math.ceil((end_min_child_weight-start_min_child_weight)/5)
    
    start_subsample = kwargs.get('start_subsample', .7)
    end_subsample = kwargs.get('end_subsample', .9)
    step_subsample = (end_subsample-start_subsample)/5
    
    # since it takes too much computational resource, disable the gamma search to reduce resource demand
    # start_gamma = kwargs.get('start_gamma', .0)
    # end_gamma = kwargs.get('end_gamma', .4)
    # step_gamma = (end_gamma-start_gamma)/5
    
    model = XGBClassifier()
    param_grid = {}
    
    eta = [round(i,4) for i in np.arange(start_eta, end_eta, step_eta)]
    param_grid.update(dict(eta=eta))
    max_depth = [i for i in np.arange(start_max_depth, end_max_depth, step_max_depth)]
    param_grid.update(dict(max_depth=max_depth))
    min_child_weight = [i for i in np.arange(start_min_child_weight, end_min_child_weight, step_min_child_weight)]
    param_grid.update(dict(min_child_weight=min_child_weight))
    subsample = [round(i,2) for i in np.arange(start_subsample, end_subsample, step_subsample)]
    param_grid.update(dict(subsample=subsample))
    
    # gamma = [round(i,2) for i in np.arange(start_gamma, end_gamma, step_gamma)]
    # param_grid.update(dict(gamma=gamma))
    
    kfold = StratifiedKFold(n_splits=len(set(y_train)), shuffle=True)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    params.update(grid_result.best_params_)
    
    plot_ = kwargs.get('plot', False)
    if plot_:
        fig, ax = plt.subplots()
        means = grid_result.cv_results_['mean_test_score']
        plt.errorbar(eta, means)
        plt.title("XGBoost learning_rate vs Log Loss")
        plt.xlabel('learning_rate')
        plt.ylabel('Log Loss')
        return params, param_grid, fig
    return params, param_grid

# @st.cache
def _plot_performance_matrix(model_result):
    # plot performance metrics
    fig, ax = plt.subplots(figsize=(12,12))
    plt.plot(model_result['validation_0'][next(iter(model_result['validation_0']))], label='train')
    plt.plot(model_result['validation_1'][next(iter(model_result['validation_1']))], label='test')
    # show the legend
    plt.legend()
    #show title
    plt.title('Performance Matrix')
    plt.xlabel('# of iteration')
    plt.ylabel('Loss')
    return fig


def auto_xgboost():
    # main
    st.title('XGBoosting')
    st.write('This app is powered by Streamlit, Sklearn.')
    st.write("""Working on Hyperparameters tuning would not fit to the model directly 
             unless user clicks the confirm button. It will take awhile for hyperparameters tuning.
             Please stay tuned while it is loading.""")
    df, uploaded = upload_file(file_type = ['csv', 'xlsx', 'xls'], show_file_info = False, key='train')
    
    if not uploaded:
        st.stop()
    
    show_upload = st.beta_expander('Preview uploaded dataset', expanded=True)
    show_upload.write(df.head(50))
    
    # select target column
    y_ = st.sidebar.selectbox('Select target variable', df.columns.to_list(),
                              help='Please choose the target label')
    features = df.columns.to_list()
    features.remove(y_)
    
    # data cleansing
    label = LabelEncoder()
    imp_freq = SimpleImputer(strategy='most_frequent')
    imp_median = SimpleImputer(strategy='median')
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    std = StandardScaler()
    
    # label encoding
    df[y_] = label.fit_transform(np.array(df[y_]).reshape(-1, 1))
    
    # standard encoding for numeric columns
    numeric_cols = df[features].select_dtypes(include='number').columns.to_list()
    if len(numeric_cols)>0:
        df[numeric_cols] = imp_median.fit_transform(df[numeric_cols])
        df[numeric_cols] = std.fit_transform(df[numeric_cols])
    
    # one hot encoding for categorical columns
    object_cols = df[features].select_dtypes(include='object').columns.to_list()
    onehot_df = pd.DataFrame()
    if len(object_cols)>0:
        df[object_cols] = imp_freq.fit_transform(df[object_cols])
        onehot_df = pd.DataFrame(data=onehot.fit_transform(df[object_cols]), columns=onehot.get_feature_names())
    
    # retrieve all cols
    df = pd.concat([df[numeric_cols], df[y_], onehot_df], axis=1)
    
    show_transform = st.beta_expander('Preview transformed dataset', expanded=False)
    show_transform.write(df.head(50))
    
    # data modeling
    model = XGBClassifier()
    x_ = df.columns.to_list()
    x_.remove(y_)
    x_train, x_valid, y_train, y_valid = train_test_split(df[x_], df[y_], train_size=.7, random_state=0)
    
    # tune model
    hyper = st.sidebar.beta_expander("Hyperparameters Tuning")
    objective_method_ = hyper.selectbox('Objective',
                                     ['reg:linear', 'reg:logistic','binary:logistic',
                                      'binary:logitraw', 'count:poisson', 'multi:softmax',
                                      'multi:softprob','rank:pairwise','reg:gamma','reg:tweedie'],
                                     index=6,
                                     help='by default multi:softprob')
    
    params = {
        # Other parameters
        # 'n_estimators': 100,
    }
    
    # hyper parameter selection 
    # col1, col2 = hyper.columns(2)    
    start_eta_ = hyper.number_input(label='Start ETA Range', value=.01, 
                                           min_value=.0, max_value=1., format="%.2f")
    end_eta_ = hyper.number_input('End ETA Range', value=.3, 
                               min_value=start_eta_, max_value=1., format="%.2f")
    
    start_max_depth_ = hyper.number_input(label='Start Max Depth Range', value= 2, 
                                           min_value=0, step=1)
    end_max_depth_ = hyper.number_input(label='End Max Depth Range', value= 12, 
                                     min_value=int(start_max_depth_), step=1)
    
    start_min_child_weight_ = hyper.number_input(label='Start Min Child Weight Range', value=0, 
                                           min_value=0, step=1)
    end_min_child_weight_ = hyper.number_input(label='End Min Child Weight Range', value= 3, 
                                           min_value=int(start_min_child_weight_), step=1)
    
    start_subsample_ = hyper.number_input(label='Start Subsample Range', value=.7, 
                                           min_value=.1, max_value=1., format="%.2f")
    end_subsample_ = hyper.number_input(label='End Subsample Range', value=.9, 
                                     min_value=start_subsample_, max_value=1., format="%.2f")
    
    # start_gamma_ = hyper.number_input(label='Start Gamma Range', value=.0, 
    #                                        min_value=.0, max_value=1., format="%.2f")
    # end_gamma_ = hyper.number_input(label='End Gamma Range', value=.4, 
    #                              min_value=start_gamma_, max_value=1., format="%.2f")
        
    hyperparameter_train = hyper.button('Confirm hyperparameters setting', 
                                        help="It will take awhile")
    if hyperparameter_train:
        params['objective'] = objective_method_
        params = _get_best_iteration(x_train, x_valid, y_train, y_valid, params)
        params, param_grid = _get_best_hyperparameters(x_train, y_train, params,
                                start_learning_rate=start_eta_,
                                end_learning_rate=end_eta_,
                                start_max_depth=start_max_depth_,
                                end_max_depth=end_max_depth_,
                                start_min_child_weight=start_min_child_weight_,
                                end_min_child_weight=end_min_child_weight_,
                                start_subsample=start_subsample_,
                                end_subsamples=end_subsample_,
                                
                                # start_gamma=start_gamma_,
                                # end_gamma=end_gamma_,
                                )
        st.markdown('*Tested hyperparameters*')
        st.write(pd.DataFrame.from_dict(param_grid))
        st.markdown('*Opitmal hyperparameters found*')
        st.session_state.params = params
        st.write(st.session_state.params)
    
    col1, col2 = st.beta_columns((1,3))
    model_start_ = col1.button('Confirm training', 
                             help='Please confirm the target variable in sidebar before training')
    show_hyparameters_ = col2.checkbox('Show model hyper-parameters')
    
    st.session_state.model = True if 'model' not in st.session_state else st.session_state.model # True for empty model
    
    if model_start_ :
        if 'params' not in st.session_state:
            st.session_state.params = {}
        model.set_params(**st.session_state.params)  
        evalset = [(x_train, y_train), (x_valid, y_valid)]
        model.fit(x_train, y_train, eval_set=evalset)
        st.session_state.model = model
        st.success('Model Completed!')
    elif (not model_start_) and (st.session_state.model == True): 
        st.stop()
    
    # retrieve model information for last trained
    model = st.session_state.model
    
    # k-fold evaluation
    with st.spinner('Model evaluation in progress'):
        kfold = StratifiedKFold(n_splits=10)
        if ('cross_val_score' not in st.session_state) or model_start_:
            results = cross_val_score(model, df[x_], df[y_], cv=kfold)
            st.session_state.cross_val_score = results
        results = st.session_state.cross_val_score # retrieve cross val result for last evaluation
        st.write("K-fold Accuracy: %.2f%% (STD: %.2f%%)" % (results.mean()*100, results.std()*100))
    
    # show xgboost model hyper-parameters
    if show_hyparameters_:
        st.write(model.get_xgb_params())
    
    st.subheader('Visualization')
    col3, col4 = st.beta_columns(2)
    # performance metrics
    if col3.checkbox('Performance metrics'):
        model_result = model.evals_result()
        col3.write(_plot_performance_matrix(model_result))
    
    # plot top 10 important features
    if col4.checkbox('Top 10 important features'):
        fig, ax = plt.subplots(figsize=(12,12))
        plot_importance(model, ax=ax, max_num_features=10)
        col4.write(fig)
    
    # plot 4 trees to reduce computational resource required
    if st.checkbox('Show tree', help='it will take quite a while to plot the decision tree'):
        with st.spinner('plotting tree'):
            fig, ax = plt.subplots(figsize=(30, 30))
            plot_tree(model, ax=ax,
                      # num_trees=4,
                      )
            st.write(fig)
    
    # predict for existing dataset
    df['y_pred'] = model.predict(df[x_])
    
    if len(numeric_cols) > 0:
        df[numeric_cols] = std.inverse_transform(df[numeric_cols])
    if len(object_cols) >0:
        df[object_cols] = onehot.inverse_transform(onehot_df)
    df[y_] = label.inverse_transform(df[y_])
    df['y_pred'] = label.inverse_transform(df['y_pred'])
    
    # Download
    show_result = st.beta_expander('Result dataframe', expanded=False)
    show_result.dataframe(df[[y_, 'y_pred'] + numeric_cols + object_cols].head(50))
    
    st.subheader('Download result and model')
    download_file(df[[y_, 'y_pred'] + numeric_cols + object_cols])
    download_model(model)
    
    
    # upload testing dataset and clean and forecast
    st.subheader('Upload testing dataset and predict result (Beta function)')
    start_test = st.checkbox('Start testing and forecasting function')
    
    if not start_test:
        st.stop()
        
    # upload new testing dataframe and predict
    df_test, uploaded_test = upload_file(file_type = ['csv', 'xlsx', 'xls'], show_file_info = False, key='test')
    if not uploaded_test:
        st.stop()
        
    show_uploaded_test = st.beta_expander('Preview uploaded dataset', expanded=True)
    show_uploaded_test.write(df_test.head(50))
    
    if y_ in df_test.columns.to_list():
        df_test[y_] = label.transform(np.array(df_test[y_]).reshape(-1, 1))
    else:
        df_test[y_] = None
    
    # check if same set of cols exist in testing set
    features_test = df_test.columns.to_list()
    features_test.remove(y_)
    exist = all([item in features for item in features_test])
    if not exist:
        st.error('There are column(s) missing in the testing set')
        st.stop()
    
    numeric_cols_test = df_test[features].select_dtypes(include='number').columns.to_list()
    if len(numeric_cols_test)>0:
        df_test[numeric_cols_test] = imp_median.transform(df_test[numeric_cols_test])
        df_test[numeric_cols_test] = std.transform(df_test[numeric_cols_test])
    
    # one hot encoding for categorical columns
    object_cols_test = df_test[features].select_dtypes(include='object').columns.to_list()
    onehot_df_test = pd.DataFrame()
    if len(object_cols_test)>0:
        df_test[object_cols_test] = imp_freq.transform(df_test[object_cols_test])
        onehot_df_test = pd.DataFrame(data=onehot.transform(df_test[object_cols_test]), 
                                      columns=onehot.get_feature_names())
    
    # retrieve all cols
    df_test = pd.concat([df_test[numeric_cols_test], df_test[y_], onehot_df_test], axis=1)
    
    show_transform_test = st.beta_expander('Preview transformed dataset', expanded=False)
    show_transform_test.write(df_test.head(50))
    
    # predict
    x_test_ = df_test.columns.to_list()
    x_test_.remove(y_)
    
    df_test['y_pred'] = model.predict(df_test[x_test_])
    
    if len(numeric_cols_test) > 0:
        df_test[numeric_cols_test] = std.inverse_transform(df_test[numeric_cols_test])
    if len(object_cols_test) >0:
        df_test[object_cols_test] = onehot.inverse_transform(onehot_df_test)
    if df_test[y_].notnull().all():
        df_test[y_] = label.inverse_transform(df_test[y_])
    df_test['y_pred'] = label.inverse_transform(df_test['y_pred'])
    
    # Download
    show_result_test = st.beta_expander('Result dataframe', expanded=False)
    show_result_test.dataframe(df_test[[y_, 'y_pred'] + numeric_cols_test + object_cols_test].head(50))
    
    st.subheader('Download result')
    download_file(df_test[[y_, 'y_pred'] + numeric_cols_test + object_cols_test])