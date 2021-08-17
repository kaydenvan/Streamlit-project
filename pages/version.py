# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 21:26:14 2021

@author: van_s
"""
import streamlit as st

def update():
    st.title('Version update record')
    st.write('This page records all the update history for this app.')
    st.markdown("""
                *17th Aug 2021*\n
                1. Add XGBoosting detail function
                2. Roll back for beta_columns and beta_expander since streamlit online version is outdated
                3. Redesign UI UX
                """)
    st.markdown("""
                *15th Aug 2021*\n
                1. Add demo function in auto clustering
                """)
    st.markdown("""
                *14th Aug 2021*\n
                1. Plot function on auto clustering
                """)
    st.markdown("""
                *12th Aug 2021*\n
                1. Minor update on categorical automl
                2. Minor update on auto clustering
                """)
    st.markdown("""
                *9th Aug 2021*\n
                1. Minor update on demo function
                """)
    st.markdown("""
                *8th Aug 2021*\n
                1. Update demo function in EDA
                2. Add demo function in categorical automl and stock auto ml
                """)
    st.markdown("""
                *6th Aug 2021*\n
                1. Disable download function
                2. Customer Auto Segmentation beta 0.1
                """)
    st.markdown("""
                *5th Aug 2021*\n
                1. Change upload file to function to increase usability
                2. Change the upload in categorical automl and eda
                """)
    st.markdown("""
                *3rd Aug 2021*\n
                1. Roll off total view count due to network issue
                2. hide message box
                """)
    st.markdown("""
                *2nd Aug 2021*\n
                1. Update on message box on main page
                2. Update page name
                3. Add reminder for stock auto machine learning
                4. Add update page
                5. Add total view count
                """)