# -*- coding: utf-8 -*-
"""
reference: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/6
"""


import streamlit as st
import base64
import io

def download(df):
    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="result.xlsx">Download excel file</a>'
    st.markdown(linko, unsafe_allow_html=True)