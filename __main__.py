# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 01:14:50 2021

@author: van_s
"""

from func.multipage import MultiPage
from pages.main import main
from pages.TimeSeries_AutoML import Stock_AutoML
from pages.exploratory_data_analysis import exploratory_data_analysis

# create instance
app = MultiPage()

# add pages
app.add_page("Main Page", main)
app.add_page('Exploratory Data Analysis', exploratory_data_analysis)
app.add_page('Stock AutoML', Stock_AutoML)

# init instance
app.run()