import streamlit as st
import pandas as pd
import sqlite3 as sq
import datetime
import yfinance as yf
from preprocess import preprocessing
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


def make_plot(data_final):
    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(go.Scatter(x = data_final.index,y = data_final.High, marker_color='blue', name='High'))
    fig.add_trace(go.Scatter(x = data_final.index,y = data_final.Low, marker_color='blue', name='Low'))
    fig.add_trace(go.Scatter(x = data_final.index,y = data_final.Forecast_High, marker_color='green', name='Predicted High'))
    fig.add_trace(go.Scatter(x = data_final.index,y = data_final.Forecast_Low, marker_color='red', name='Predicted Low'))
    fig.update_yaxes(range=[0,700000000],secondary_y=True)
    fig.update_yaxes(visible=False, secondary_y=True)
    fig.update_layout(title={'text':choice_name, 'x':0.5}, width = 800, height = 500, plot_bgcolor = 'rgba(0,0,0,0)')
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig


"# Stock Price Prediction"
"This is an app to predict the High and Low of the given Stock. You can select different stocks, intervals, periods from the sidebar. Feel free to experiment with different models"

db = sq.connect('stocks.db')

# get country
query = "SELECT DISTINCT(Country) FROM tkrinfo;"
country = pd.read_sql_query(query, db)
choice_country = st.sidebar.selectbox("Pick country", country)

# get exchange
query = "SELECT DISTINCT(Exchange) FROM tkrinfo WHERE Country = '" + choice_country + "'"
exchange = pd.read_sql_query(query, db)
choice_exchange = st.sidebar.selectbox("Pick exchange", exchange, index = 1)

# get stock name
query = "SELECT DISTINCT(Name) FROM tkrinfo WHERE Exchange = '" + choice_exchange + "'"
name = pd.read_sql_query(query, db)
choice_name = st.sidebar.selectbox("Pick the Stock", name)

# get stock tickr
query = "SELECT DISTINCT(Ticker) FROM tkrinfo WHERE Exchange = '" + choice_exchange + "'" + "and Name = '" + choice_name + "'"
ticker_name = pd.read_sql_query(query, db)
ticker_name = ticker_name.loc[0][0]


# get interval
interval = st.sidebar.selectbox("Interval", ['1d', '1wk', '1mo', '3mo'])

#get period
period = st.sidebar.selectbox("Period",['1mo','3mo','6mo','1y','2y','5y','10y','max'],index = 2)

# get stock data
stock = yf.Ticker(str(ticker_name))

data = stock.history(interval=interval, period=period)

if len(data)==0:
    st.write("Unable to retrieve data.This ticker may no longer be in use. Try some other stock")
else:

    #preprocessing
    data = preprocessing(data,interval)

    if period == '1mo' or period == '3mo':
        horizon = st.sidebar.slider("Forecast horizon",1,15,5)
    else:
        if interval == '1d' or interval == '1wk':
            horizon = st.sidebar.slider("Forecast horizon", 1, 30, 5)
        else:
            horizon = st.sidebar.slider("Forecast horizon", 1, 15, 5)

    model = st.selectbox('Model',['Simple Exponential Smoothing','Halt Model','Holt-Winter Model','Auto Regressive Model',
                                  'Moving Average Model','ARMA Model', 'ARIMA Model','AutoARIMA',
                                  'Linear Regression','Random Forest', 'Gradient Boosting','Support Vector Machines',
                                  ])

    if model=='Simple Exponential Smoothing':
        col1,col2 = st.columns(2)
        with col1:
            alpha_high = st.slider("Alpha_high",0.0,1.0,0.20)
        with col2:
            alpha_low = st.slider("Alpha_low",0.0,1.0,0.25)
        from SES import SES_model
        data_final, smap_low, smap_high, optim_alpha_high, optim_alpha_low = SES_model(data,horizon,alpha_high,alpha_low)

#data_final

        #st.line_chart(data_final[['High','Forecast_High','Low','Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

    elif model == 'Halt Model':
        col1, col2,col3,col4 = st.columns(4)
        with col1:
            level_high = st.slider("Level High", 0.0, 1.0, 0.20)
        with col2:
            trend_high = st.slider("Trend high", 0.0, 1.0, 0.20)
        with col3:
            level_low = st.slider("Level low", 0.0, 1.0, 0.20)
        with col4:
            trend_low = st.slider("Trend Low", 0.0, 1.0, 0.20)
        from SES import Holt_model
        data_final,smap_low,smap_high,optim_level_high,optim_level_low,optim_trend_high,optim_trend_low = Holt_model(data,horizon
                                                                        ,level_high,level_low,trend_high,trend_low)
        
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])

        

        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))


    elif model == 'Holt-Winter Model':
        col1, col2 = st.columns(2)
        with col1:
            level_high = st.slider("Level High", 0.0, 1.0, 0.20)
            trend_high = st.slider("Trend high", 0.0, 1.0, 0.20)
            season_high = st.slider("Seasonal high", 0.0, 1.0, 0.20)
        with col2:
            level_low = st.slider("Level low", 0.0, 1.0, 0.20)
            trend_low = st.slider("Trend Low", 0.0, 1.0, 0.20)
            season_low = st.slider("Seasonal Low", 0.0, 1.0, 0.20)
        from SES import Holt_Winter_Model
        data_final, smap_low, smap_high, optim_level_high, optim_level_low, optim_trend_high, optim_trend_low, optim_season_high, optim_season_low = Holt_Winter_Model(data,horizon, level_high, level_low,trend_high,trend_low,season_high,season_low)

        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1, col2 = st.columns(2)
        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

    elif model == 'Auto Regressive Model':
        col1, col2 = st.columns(2)
        with col1:
            p_high = st.slider("Order of High", 1, 30, 1)
        with col2:
            p_low = st.slider("Order of Low", 1, 30, 1)
        from SES import AR_model

        data_final, smap_high, smap_low = AR_model(data,horizon,p_high,p_low)
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

    elif model == 'Moving Average Model':
        col1, col2 = st.columns(2)
        with col1:
            q_high = st.slider("Order of High", 1, 30, 1)
        with col2:
            q_low = st.slider("Order of Low", 1, 30, 1)
        from SES import AR_model
        data_final, smap_high, smap_low = AR_model(data, horizon, q_high, q_low)
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

    elif model == 'ARMA Model':
        col1, col2 = st.columns(2)
        with col1:
            p_high = st.slider("Order of AR High", 1, 30, 1)
            q_high = st.slider("Order of MA High", 1, 30, 1)
        with col2:
            p_low = st.slider("Order of AR Low", 1, 30, 1)
            q_low = st.slider("Order of MA Low", 1, 30, 1)
        from SES import ARMA_model
        data_final, smap_high, smap_low = ARMA_model(data,horizon,p_high,p_low,q_high,q_low)
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

    elif model == 'ARIMA Model':
        col1, col2 = st.columns(2)
        with col1:
            p_high = st.slider("Order of AR High", 1, 30, 1)
            q_high = st.slider("Order of MA High", 1, 30, 1)
            i_high = st.slider("Order of Differencing High" , 0,10,0)
        with col2:
            p_low = st.slider("Order of AR Low", 1, 30, 1)
            q_low = st.slider("Order of MA Low", 1, 30, 1)
            i_low = st.slider("Order of Differencing Low", 0, 10, 0)
        from SES import ARIMA_model
        data_final, smap_high, smap_low = ARIMA_model(data,horizon,p_high,p_low,q_high,q_low,i_high,i_low)
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

    elif model == 'AutoARIMA':
        from SES import Auto_Arima
        st.write("Note: This model may take some time to fit")
        data_final = Auto_Arima(data,horizon)
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))


    else:
        from ML_models import forecast
        #data_final = forecast(data,horizon,model)
        data_final, smape_high, smape_low = forecast(data,horizon,model)
        #st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        st.plotly_chart(make_plot(data_final))

        col1,col2 = st.columns(2)
        with col1:
            st.write("Average of Predicted High is : {}".format(data_final.Forecast_High.mean()))
        with col2:
            st.write("Average of Predicted Low is : {} ".format(data_final.Forecast_Low.mean()))

db.close()
