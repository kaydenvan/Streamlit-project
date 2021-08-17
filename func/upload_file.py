# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:14:02 2021

@author: van_s
"""

import pandas as pd
import streamlit as st

def upload_file(**kwargs):
    file_type = kwargs.get('file_type', ['csv', 'xlsx', 'xls'])
    file_info = kwargs.get('show_file_info', False)
    file_key = kwargs.get('key', 'upload_file')
    return_file_name = kwargs.get('return_file_name', False)
    uploaded_file = st.file_uploader('Please upload your dataset', type=file_type, key=file_key)
    file_name = ''
    if uploaded_file is not None:
        upload_status = True
        file_name = uploaded_file.name
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
    
    if return_file_name:
        return df, upload_status, file_name
    
    return df, upload_status