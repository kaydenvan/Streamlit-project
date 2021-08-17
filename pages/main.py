# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:00:03 2021

@author: van_s
"""
import streamlit as st

def display_massage(bucket=st):
    bucket.subheader("Message on 29 July 2021")
    bucket.markdown("""
                
                """)
    bucket.write()  
    
@st.cache(allow_output_mutation=True)
def Pageviews():
    return []

def main():
    st.title('Main Page')
    # msg = st.beta_expander('Message', expanded=True)
    # display_massage(msg)
    
    st.markdown("Description:")
    st.markdown("""This webstie is created for demonstration purpose by *Kayden Van*.
                If you see any bugs/errors, feel free to contact me for more enhancement.
                Besides, you may click the app navigation bar to select different functions.
                """)
    st.write()
    st.markdown("""**Current/Aimed Functions available**:<br>
                1. Exploratory Data Analysis<br>
                2. Categorical Prediction (Auto)<br>
                3. Stock Time Series Prediction (Auto)<br>
                4. Customer Segmentation (Auto)<br>
                5. XGBoost classification model with detail analysis (Beta)<br>
                6. Sentiment Analysis (Not Yet Developed)<br>
                7. Image Classification (Not Yet Developed)<br>
                """, unsafe_allow_html=True)
    
    # count page views
    if 'view' not in st.session_state:
        st.session_state.view = 1
    else:
        st.session_state.view += 1
    
    st.markdown(f'Total view: {st.session_status.view}')
    
