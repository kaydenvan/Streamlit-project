# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:12:50 2021

@author: van_s
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# st.set_option('deprecation.showPyplotGlobalUse', False)

def _univariate_numeric(df, col):
    df[col] = df[col].astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    sns.histplot(data=df, x=col, ax=ax[0]).set_title(col + '_bar')
    sns.kdeplot(data=df, x=col, ax=ax[1]).set_title(col + '_kde')
    sns.boxplot(data=df, y=col, ax=ax[2]).set_title(col + '_box')
    st.pyplot(fig)
    
def _univariate_object(df, col):
    fig, ax = plt.subplots()
    ax = sns.countplot(data=df, x=col)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
    st.pyplot(fig)

def corr(df, correlation = .7):
    col = df.select_dtypes(include='number').columns
    _ = df[col].corr().unstack().sort_values(ascending=False).drop_duplicates().reset_index()
    _ = _[_['level_0'] != _['level_1']]
    return _[(_[0] >= correlation) | (_[0] <= -correlation)].reset_index(drop=True)
    
def describe_corr(df):
    if df.empty:
        st.write('The columns are with low correlation between each other')
    else:
        for index, item in df.iterrows():
            if item[0] > 0:
                st.markdown(f"*{item['level_0'].title()}* has **positive** correlation({item[0]:.2f}) with *{item['level_1'].title()}*.")
            elif item[0] < 0:
                st.markdown(f"*{item['level_0'].title()}* has **negative** correlation({item[0]:.2f}) with *{item['level_1'].title()}*.")
            else:
                st.markdown(f"*{item['level_0'].title()}* has **no** relationship with *{item['level_1'].title()}*")

def color_df(val):
    color = 'green' if val > 0 else 'red'
    return f'background-color: {color}'

def plot_corr(df):
    fig, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(df.select_dtypes(include='number').corr(),
               annot=True, square=True, ax=ax, vmin=-1, vmax=1)
    st.pyplot(fig)

def exploratory_data_analysis():
    st.title('Exploratory Data Analysis')
    st.write('This app is powered by Matplotlib and Seaborn and aims to allow auto EDA process')
    uploaded_file = st.file_uploader('Please upload your dataset', type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
    else:
        st.stop()
    
    # preview uploaded dataframe
    st.write('Preview uploaded dataframe')
    if uploaded_file.name.split('.')[-1] in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
    elif uploaded_file.name.split('.')[-1]  == 'csv':
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
    rows, cols_no = df.shape[0], df.shape[1]
    cols = ', '.join(df.columns)
    st.write(f'The dataframe contains {rows} records and {cols_no} columns')
    st.write(f'Columns contain in dataframe: {cols}')
    st.write()
    
    # generate highlevel description
    if st.checkbox('Overall description of the dataframe'):
        st.dataframe(df.describe(include=['number', 'object']))

    
    # see if any null field
    if st.checkbox('Number of null field in each column'):
        st.dataframe(pd.DataFrame(df.isna().sum()).transpose())
        
    # categorize column type
    if st.checkbox('Categorize columns type'):
        st.markdown(f"*Numeric Cols*: {', '.join(df.select_dtypes(include=['number']).columns.values)}")
        st.markdown(f"*Object Cols*: {', '.join(df.select_dtypes(include=['object']).columns.values)}")
        st.markdown(f"*Datetime Cols*: {', '.join(df.select_dtypes(include=['datetime64']).columns.values)}")
        st.markdown(f"*Boolean Cols*: {', '.join(df.select_dtypes(include=['bool']).columns.values)}")
        
    # visualize columns
    if st.checkbox('Visualize column behavior'):
        with st.spinner('Rendering in process'):
            if len(df.select_dtypes(include=['object']).columns)>0:
                st.write('Plot Object Columns')
                for col in df.select_dtypes(include=['object']).columns:
                    _univariate_object(df, col)
            if len(df.select_dtypes(include=['number']).columns)>0:
                st.write('Plot Numeric Columns')
                for col in df.select_dtypes(include=['number']).columns:
                    _univariate_numeric(df, col)

    # show data correlation
    if st.checkbox('Show data correlation'):
        corr_factor = .7
        corr_factor = st.slider('Show variables with +-{0} correlation'.format(corr_factor), 
                                min_value=0., max_value=1., value=corr_factor)
        corr_df = corr(df, corr_factor)
        st.dataframe(corr_df.style.applymap(color_df, subset=[0]))
        # describe_corr(corr_df)
        st.markdown('**Correlation chart for all variables within the dataset**')
        with st.spinner('Rendering in process'):
            plot_corr(df)