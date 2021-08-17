# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:14:02 2021

@author: van_s
"""

import pandas as pd
import streamlit as st

def upload_file(**kwargs):
    file_type = kwargs.get('file_type', ['csv', 'xlsx', 'xls'])
    file_info = kwargs.get('show_file_info', True)
    file_key = kwargs.get('key', 'upload_file')
    uploaded_file = st.file_uploader('Please upload your dataset', type=file_type, key=file_key)
    if uploaded_file is not None:
        upload_status = True
        if file_info:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            st.write(file_details)
    else: 
        upload_status = False
    
    if upload_status:
        if uploaded_file.name.split('.')[-1] in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.split('.')[-1]  == 'csv':
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame()
    
    return df, upload_status