# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:12:50 2021

@author: van_s
"""
import streamlit as st
import pandas as pd

def exploratory_data_analysis():
    st.title('Exploratory Data Analysis')
    st.write('This page aims to allow auto EDA process')
    uploaded_file = st.file_uploader('Please upload your dataset', type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
    
    # preview uploaded dataframe
    st.write('Preview uploaded dataframe')
    if uploaded_file.name.split('.')[-1] in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
    elif uploaded_file.name.split('.')[-1]  == 'csv':
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())