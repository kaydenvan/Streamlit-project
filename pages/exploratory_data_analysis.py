# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:12:50 2021

@author: van_s
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from func.upload_file import upload_file

@st.cache
def iris_dataset():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    return df

# @st.cache(suppress_st_warning=True)
def _univariate_numeric(df, col, bucket = st):
    df[col] = df[col].astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    sns.histplot(data=df, x=col, ax=ax[0]).set_title(col + '_bar')
    sns.kdeplot(data=df, x=col, ax=ax[1]).set_title(col + '_kde')
    sns.boxplot(data=df, y=col, ax=ax[2]).set_title(col + '_box')
    bucket.pyplot(fig)
    
# @st.cache(suppress_st_warning=True)
def _univariate_object(df, col, bucket = st):
    fig, ax = plt.subplots()
    ax = sns.countplot(data=df, x=col)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
    bucket.pyplot(fig)

@st.cache(suppress_st_warning=True)
def corr(df, correlation = .7):
    col = df.select_dtypes(include='number').columns
    _ = df[col].corr().unstack().sort_values(ascending=False).drop_duplicates().reset_index()
    _ = _[_['level_0'] != _['level_1']]
    return _[(_[0] >= correlation) | (_[0] <= -correlation)].reset_index(drop=True)
    
# @st.cache(suppress_st_warning=True)
def describe_corr(df, bucket=st):
    if df.empty:
        bucket.write('The columns are with low correlation between each other')
    else:
        for index, item in df.iterrows():
            if item[0] > 0:
                bucket.markdown(f"*{item['level_0'].title()}* has **positive** correlation({item[0]:.2f}) with *{item['level_1'].title()}*.")
            elif item[0] < 0:
                bucket.markdown(f"*{item['level_0'].title()}* has **negative** correlation({item[0]:.2f}) with *{item['level_1'].title()}*.")
            else:
                bucket.markdown(f"*{item['level_0'].title()}* has **no** relationship with *{item['level_1'].title()}*")

def color_df(val):
    color = 'green' if val > 0 else 'red'
    return f'background-color: {color}'

def plot_corr(df, bucket=st):
    fig, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(df.select_dtypes(include='number').corr(),
               annot=True, square=True, ax=ax, vmin=-1, vmax=1)
    return fig

def exploratory_data_analysis():
    st.title('Exploratory Data Analysis')
    st.write('This app is powered by Matplotlib and Seaborn and aims to allow auto EDA process')
    st.markdown("""Please upload the dataset you would like to explore. 
                If you would like to see the demo, please check on the demo button.""")
    st.markdown("""*Remark: for simplicity, it is now not supported for configurating to specify the row number of column header.
                Let me know if it is an important feature if necessary.*""")
    df = pd.DataFrame()
    
    df, uploaded = upload_file(file_type = ['csv', 'xlsx', 'xls'], show_file_info = True)
    if not uploaded:
        demo = st.sidebar.radio('Enable Demo', ('Yes', 'No'), index=1,
                                help='Iris dataset is used for demonstration purpose')
        if demo == 'Yes':
            df = iris_dataset()
    else:
        demo = 'No'
    
    # if dataframe is empty, stop program
    if df.empty:
        st.stop()
    
    show_upload = st.beta_expander('Preview uploaded dataframe', expanded=True) if uploaded else st.beta_expander('Preview demo dataframe', expanded=True)
    show_upload.dataframe(df.head(50))
            
    rows, cols_no = df.shape[0], df.shape[1]
    cols = ', '.join(df.columns)
    st.write(f'The dataframe contains {rows} records and {cols_no} columns')
    st.write(f'Columns contain in dataframe: {cols}')
    st.write()
    
    # generate highlevel description
    show_summary = st.beta_expander('Summary of dataframe', expanded=False)
    show_summary.dataframe(df.describe())

    # see if any null field
    st.subheader('Number of null field in each column')
    null_df = pd.DataFrame(df.isna().sum().loc[lambda x : x>0].sort_values(ascending=False))
    if not null_df.empty:
        st.dataframe(null_df.transpose())
    else:
         
        st.write('No null column found')
        
    # categorize column type
    st.subheader('Categorize columns type')
    st.markdown(f"*Numeric Cols*: {', '.join(df.select_dtypes(include=['number']).columns.values)}")
    st.markdown(f"*Object Cols*: {', '.join(df.select_dtypes(include=['object']).columns.values)}")
    st.markdown(f"*Datetime Cols*: {', '.join(df.select_dtypes(include=['datetime64']).columns.values)}")
    st.markdown(f"*Boolean Cols*: {', '.join(df.select_dtypes(include=['bool']).columns.values)}")
        
    # enable plot button only not in demo mode
    show_plot_cols = st.beta_expander('Visualize column behavior', expanded=False)\
        if demo == 'No' else st.beta_expander('Visualize column behavior', expanded=True)
    
    # visualize columns
    if len(df.select_dtypes(include=['object']).columns)>0:
        show_plot_cols.write('Plot Object Columns')
        for col in df.select_dtypes(include=['object']).columns:
            _univariate_object(df, col, bucket=show_plot_cols)
    if len(df.select_dtypes(include=['number']).columns)>0:
        show_plot_cols.write('Plot Numeric Columns')
        for col in df.select_dtypes(include=['number']).columns:
            _univariate_numeric(df, col, bucket=show_plot_cols)

    show_corr = st.beta_expander('Data Correlation', expanded=False)\
        if demo == 'No' else st.beta_expander('Data Correlation', expanded=True)

    # show data correlation
    corr_factor = .7
    corr_factor = show_corr.slider('Configuration: Show variables with +-{0} correlation'.format(corr_factor), 
                            min_value=0., max_value=1., value=.7) if demo == 'No' else .7
    corr_df = corr(df, corr_factor)
    show_corr.dataframe(corr_df)
    # st.dataframe(corr_df.style.applymap(color_df, subset=[0])) # there is bug on the style
    describe_corr(corr_df, show_corr)
    show_corr.markdown('**Correlation chart for all variables within the dataset**')
    show_corr.write(plot_corr(df, show_corr))