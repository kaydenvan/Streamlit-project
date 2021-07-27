# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:31:27 2021

@author: van_s
"""

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

def stock_automl():
    st.title('Time Series AutoML')
    st.write('This app is powered by Streamlit, Yahoo Finance and FbProphet')
    stock_symbol = st.text_input('Please input stock symbol', '').strip()
    if len(stock_symbol.strip()) != 0:
        df = yf.Ticker(stock_symbol).history(period='10y')
        df.reset_index(inplace=True, drop=False)
        if df.empty:
            st.error('stock symbol is invalid')  
            st.stop()
        else:
            st.write(f'{stock_symbol.upper()} historical record is retriving')
            st.write(f'Stock record from {min(df.Date).date()} to {max(df.Date).date()}')
    # elif stock_symbol == ' ':
    #     st.stop()
    else: 
    #     st.error('Empty stock symbol')
        st.stop()
    
    # create selection button
    button_details = dict(count=1,label='1m',step='month',stepmode='backward'),\
                     dict(count=6,label='6m',step='month',stepmode='backward'),\
                     dict(count=1,label='YTD',step='year',stepmode='todate'),\
                     dict(count=1,label='1y',step='year',stepmode='backward'),\
                     dict(count=3,label='3y',step='year',stepmode='backward'),\
                     dict(step='all')
    
    fig = go.Figure(data = go.Ohlc(
            x=df['Date'],\
            open=df['Open'],\
            high=df['High'],\
            close=df['Close'],\
            low=df['Low']))
    fig.update_layout(
        title_text=f'Interactive Time Series Graph for {stock_symbol.upper()}')
    fig.update_layout(
        xaxis=dict(rangeselector=dict(
            buttons=list(button_details)),
            rangeslider=dict(visible=True),
            type='date'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.write('We are now in process of building time series model')
    
    st.title('Data Modeling')
    #data modeling
    from fbprophet import Prophet
    max_predict_day = 60
    # feature_list = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
    model = Prophet(daily_seasonality=True)
    # for feature in feature_list:
    #     model.add_regressor(feature)
    with st.spinner('Wait for model development'):
        model.fit(df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'}))
    st.success('Model created')
    
    if st.checkbox('Do you want to forecast the stock price?'):
        predict_days = st.slider('Select the number of days you want to predict',
                         min_value=7, max_value=max_predict_day)
        with st.spinner('Wait for model forecast'):
            future = model.make_future_dataframe(max_predict_day, freq='D')
            #future = pd.DataFrame(pd.bdate_range(start=max(df.Date), periods=60, freq='D'), columns=['ds'])
            forecast = model.predict(future)
            
            # plot graph
            fig, ax = plt.subplots()
            forecast_plot_df = forecast.set_index('ds')[['yhat']][-(60+max_predict_day-predict_days):-(max_predict_day-predict_days)]
            actual_plot_df = df.set_index('Date')[['Close']][(df[df['Date'] == min(forecast_plot_df.index)].index.values[0]):]
            ax.plot(forecast_plot_df, color='r', label='forecast')
            ax.plot(actual_plot_df, color='g', label='actual')
            ax.legend()
            plt.xticks(rotation=45)
            st.write(fig)
            
            # summary
            changes = ((forecast_plot_df.iloc[-1] - forecast_plot_df.iloc[0])/forecast_plot_df.iloc[0])[0]
            color = 'green' if changes > 0 else 'black' if changes == 0 else 'red'
            st.markdown(f"<font color='{color}'>**Quick Summary**: With {predict_days} days prediction, {stock_symbol} is expected to have {changes*100:.2f}% changes.</font>", 
                        unsafe_allow_html=True)
            
    st.title('Further information')
    # optional graph
    if st.checkbox('Show prediction dataframe'):
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(predict_days))
    if st.checkbox('Show components in the Time Series Model'):
        fig = model.plot_components(forecast[:-(max_predict_day-predict_days)])
        st.write(fig)




















