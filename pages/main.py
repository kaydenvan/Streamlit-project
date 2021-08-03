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
                4. Customer Segmentation (Auto) (Not Yet Developed)<br>
                5. One classification model detail analysis (Not Yet Developed)<br>
                    (e.g. pick one model, 
                     train model, 
                     plot feature importance,
                     plot criteria, 
                     allow upload to predict new data)<br>
                6. Sentiment Analysis (Not Yet Developed)
                7. Image Classification (Not Yet Developed)
                """, unsafe_allow_html=True)
    
    # count page views
    # pageviews=Pageviews()
    # pageviews.append('dummy')
    
    # try:
    #     view = '<p style="text-align: right; font-size: 8px;">Total page viewed = {} times.</p>'.format(len(pageviews))
    # except ValueError:
    #     view = '<p style="text-align: right; font-size: 8px;">Page viewed = {} times</p>.'.format(1)
    
    # st.write(view, unsafe_allow_html=True)
    # st.markdown("""You may aware that some of the functions are locked. 
    #             If you are interested on it, please contact Kayden for the information""")
    
