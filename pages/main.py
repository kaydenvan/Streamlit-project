# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:00:03 2021

@author: van_s
"""
import streamlit as st

def main():
    st.title('Main Page')
    st.write('Description:')
    st.write('This webstie is created for demonstration purpose by Kayden Van')
    st.write('If you see any bugs/error, feel free to contact me for more enhancement')
    st.write('You may click the app navigation bar to select different functions')
    st.write()
    st.markdown("""**Current Functions available**:<br>
                1. Exploratory data analysis<br>
                2. Categorical Auto Machine Learning<br>
                3. Stock Time Series Prediction<br>
                """, unsafe_allow_html=True)
    # st.markdown("""You may aware that some of the functions are locked. 
    #             If you are interested on it, please contact Kayden for the information""")
    
