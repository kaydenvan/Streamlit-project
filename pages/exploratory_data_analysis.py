# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:12:50 2021

@author: van_s
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Exploratory Data Analysis",
    layout="centered",
    initial_sidebar_state="auto")

@st.cache
def iris_dataset():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
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
    bucket.pyplot(fig)

def exploratory_data_analysis():
    st.title('Exploratory Data Analysis')
    st.write('This app is powered by Matplotlib and Seaborn and aims to allow auto EDA process')
    st.markdown("""Please upload the dataset you would like to explore. 
                If you would like to see the demo, please check on the demo button.""")
    st.markdown("""*Remark: for simplicity, it is now not supported for configurating to specify the row number of column header.
                Let me know if it is an important feature if necessary.*""")
    uploaded = False
    df = pd.DataFrame()
    col1, col2 = st.beta_columns((3, 1))
    
    with col1:
        uploaded_file = st.file_uploader('Please upload your dataset', type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            st.write(file_details)
            uploaded = True
    
    with col2: 
        if not uploaded:
            if st.checkbox('demo', help='Iris data will be used if you check this option'):
                df = iris_dataset()
     
    # preview uploaded dataframe
    if uploaded:
        if uploaded_file.name.split('.')[-1] in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.split('.')[-1]  == 'csv':
            df = pd.read_csv(uploaded_file)
    
    # if dataframe is empty, stop program
    if df.empty:
        st.stop()
    
    st.subheader('Preview uploaded dataframe') if uploaded else st.subheader('Preview demo dataframe')
    st.dataframe(df.head())
            
    rows, cols_no = df.shape[0], df.shape[1]
    cols = ', '.join(df.columns)
    st.write(f'The dataframe contains {rows} records and {cols_no} columns')
    st.write(f'Columns contain in dataframe: {cols}')
    st.write()
    
    # generate highlevel description
    st.subheader('Summary of the dataframe')
    st.dataframe(df.describe(include=['number', 'object']))

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
    
    # we don't use expander at the moment since it is by default will run everything
    # optional_1 = st.beta_expander("Optional Functions: Visualize column behavior", False)
    
    # visualize columns
    st.checkbox('Optional Functions: Visualize column behavior', key='plot_cols')
    
    if st.session_state.plot_cols:
        if len(df.select_dtypes(include=['object']).columns)>0:
            st.write('Plot Object Columns')
            for col in df.select_dtypes(include=['object']).columns:
                _univariate_object(df, col, bucket=st)
        if len(df.select_dtypes(include=['number']).columns)>0:
            st.write('Plot Numeric Columns')
            for col in df.select_dtypes(include=['number']).columns:
                _univariate_numeric(df, col, bucket=st)

    # optional_2 = st.beta_expander("Optional Functions: Data Correlation", False)
    
    # show data correlation
    st.checkbox('Optional Functions: Data Correlation', key='get_corr')
    
    if st.session_state.get_corr:
        corr_factor = .7
        corr_factor = st.slider('Show variables with +-{0} correlation'.format(corr_factor), 
                                min_value=0., max_value=1., value=corr_factor)
        corr_df = corr(df, corr_factor)
        st.dataframe(corr_df)
        # st.dataframe(corr_df.style.applymap(color_df, subset=[0])) # there is bug on the style
        describe_corr(corr_df, st)
        st.markdown('**Correlation chart for all variables within the dataset**')
        plot_corr(df, st)