# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:00:03 2021

@author: van_s
"""
import streamlit as st

def main():
    st.title('Main Page')
    st.markdown("""Description:\n
                This webstie is created for demonstration purpose by Kayden Van.\n
                If you see any bugs/errors, feel free to contact me for more enhancement.\n
                You may click the app navigation bar to select different functions
                """)
    st.write()
    st.markdown("""**Current Functions available**:<br>
                1. Exploratory Data Analysis<br>
                2. Categorical Prediction (Auto)<br>
                3. Stock Time Series Prediction (Auto)<br>
                4. Customer Segmentation (Auto) (Not Yet Developed)<br>
                5. One classification model detail analysis (Not Yet Developed)<br>
                    (e.g. pick one model, 
                     train model, 
                     plot feature importance,
                     plot criteria, 
                     allow upload to predict new data)
                """, unsafe_allow_html=True)
    # st.markdown("""You may aware that some of the functions are locked. 
    #             If you are interested on it, please contact Kayden for the information""")
    
