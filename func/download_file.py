# -*- coding: utf-8 -*-
"""
reference: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/6
"""


import streamlit as st
import pickle
import base64
import io

def download_file(df):
    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="result.xlsx">Download Excel File</a>'
    st.markdown(linko, unsafe_allow_html=True)
    
def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="trained_model.pkl">Download Trained Model File</a>'
    st.markdown(href, unsafe_allow_html=True)