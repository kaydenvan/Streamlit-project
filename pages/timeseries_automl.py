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

@st.cache
def prophet_model(df, date, target, **kwargs):
    from fbprophet import Prophet
    daily_seasonality = kwargs.get('daily_seasonality', True)
    model = Prophet(daily_seasonality=daily_seasonality)
    model.fit(df[[date, target]].rename(columns={date:'ds', target:'y'}))
    return model

@st.cache
def forecast_model(model, period, **kwargs):
    freq = kwargs.get('freq', 'D')
    future = model.make_future_dataframe(period, freq=freq)
    forecast = model.predict(future)
    return forecast

# @st.cache(allow_output_mutation=True)
# def get_stock_info(stock_symbol, **kwargs):
#     period = kwargs.get('period', '3y')
#     return yf.Ticker(stock_symbol).history(period)

def stock_automl():
    st.title('Stock Performance AutoML')
    st.write('This app is powered by Streamlit, Yahoo Finance and FbProphet')
    st.markdown("""With the limited resource on streamlit free tier, 
                it is developed with purpose that the low accuracy on the time series prediction.""")
    st.markdown("This app currently supports US stock only.")
    
    demo = st.sidebar.radio('Enable Demo', ('Yes', 'No'), index=1, 
                            help='AAPL will be used as the demo and by default 5-year records used')
    
    col1, col2 = st.beta_columns(2)
    with col1:
        stock_symbol = st.text_input('Please input stock symbol', '').strip() if demo == 'No' else 'AAPL'
    
    with col2:
        period_ = st.selectbox('How long would you extract?',
                               ('1y','2y','5y','10y','ytd','max'),
                               index=2) if demo == 'No' else '5y'
    
    if len(stock_symbol.strip()) != 0:
        df = yf.Ticker(stock_symbol).history(period=period_)
        df.reset_index(inplace=True, drop=False)
        if df.empty:
            st.error('stock symbol is invalid')  
            st.stop()
        else:
            st.write(f'{stock_symbol.upper()} {period_.upper()} historical record is retriving')
            st.write(f'Stock record from {min(df.Date).date()} to {max(df.Date).date()}')
    else: 
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
    # from fbprophet import Prophet
    # model = Prophet(daily_seasonality=True)
    max_predict_day = 60

    with st.spinner('model develop in progress'):
        model = prophet_model(df, 'Date', 'Close', daily_seasonality=True)
        # st.success('Model created')
    

    predict_days = st.slider('Select the number of days you want to predict',
                     min_value=7, max_value=max_predict_day,value=30, step=1)\
        if demo == 'No' else 30
    with st.spinner('Wait for model forecast'):
        forecast = forecast_model(model, max_predict_day, freq='D')
        
        # plot graph
        fig, ax = plt.subplots()
        if max_predict_day != predict_days:
            forecast_plot_df = forecast.set_index('ds')[['yhat']][-(60+max_predict_day-predict_days):-(max_predict_day-predict_days)]
        else:
            forecast_plot_df = forecast.set_index('ds')[['yhat']][-(60+max_predict_day-predict_days+1):]
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
        
    # optional graph
    optionals = st.beta_expander("Optional Functions", False) if demo == 'No' else st.beta_expander("Optional Functions", True)
    optionals.markdown('**Prediction dataframe**')
    optionals.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(max_predict_day))
    optionals.markdown('**Model components**')
    fig = model.plot_components(forecast[-180:])
    optionals.write(fig)




















