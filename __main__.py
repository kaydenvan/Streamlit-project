# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:14:50 2021

@author: van_s
"""
# import streamlit as st
from func.multipage import MultiPage
from pages.main import main
from pages.timeseries_automl import stock_automl
from pages.exploratory_data_analysis import exploratory_data_analysis
from pages.categorical_automl import categorical_automl

# create instance
app = MultiPage()

# bar = st.sidebar
# account = bar.text_input('Account Name')
# password = bar.text_input('Password')

# if account == st.secrets['db_username'] and password == st.secrets['db_password']:
#     unlock = True
# else:
#     unlock = False

# add pages
app.add_page("Main Page", main)
app.add_page('Exploratory Data Analysis', exploratory_data_analysis)
# if unlock == True:
app.add_page('Stock AutoML', stock_automl)
app.add_page('Categorical AutoML', categorical_automl)


# init instance
app.run()