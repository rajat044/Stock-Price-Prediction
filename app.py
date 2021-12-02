import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
#import matplotlib
#from matplotlib.backends.backend_agg import RendererAg
import datetime
import time
key = 'WDEQU9QPGN85ZOWL'

#matplotlib.use("agg")
#_lock = RendererAgg.loc

def know_symbol():

	st.sidebar.header("Don't know desired stock symbol")
	st.sidebar.subheader('Use me')

	keyword = st.sidebar.text_input("Type Stock name(without spaces)", )
	url = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords='+keyword+'&apikey='+key+'&datatype=csv'
	
	try:
		sug = pd.read_csv(url)
		sn_dict = [[n, r] for n, r in zip(sug.name, sug.region)]

		option = st.sidebar.selectbox('Select the desired one', sn_dict)
		sym = list(sug[sug.name == option[0]].symbol)[0]
		st.sidebar.write('Its symbol is:    ', sym)
	except: pass
know_symbol()

def display_spinner(text, ti):
	with st.spinner(text):
		time.sleep(ti)
	#st.success('Done!')

st.header('Enter the stock symbol')
symbol = st.text_input('')


if not symbol: st.stop()

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+symbol+'&apikey='+key+'&datatype=csv&outputsize=full'
df = pd.read_csv(url, index_col = 0, parse_dates = True)
display_spinner('Loading Data...', 2)

if not df.empty:
	col1, col2 = st.columns([1, 1])
	ds = col1.date_input(
    	"Choose the starting date",
    	datetime.date(2021, 1, 1))
	de = col2.date_input(
		"Choose the ending date",
		datetime.date(2021, 10, 10))
	
	df = df.loc[de: ds]

	fig = go.Figure()
	fig.add_trace(go.Scatter(x = df.index, y = df.open, name = 'Open', mode = 'lines'))
	fig.add_trace(go.Scatter(x = df.index, y = df.close, name = 'Close', mode = 'lines'))
	fig.add_trace(go.Scatter(x = df.index, y = df.high, name = 'High', mode = 'markers'))
	fig.add_trace(go.Scatter(x = df.index, y = df.low, name = 'Low', mode = 'markers'))
	#fig.update_traces(hoverinfo = 'text+name', mode = 'lines+markers')
	
	fig.update_xaxes(title_text = 'Date', rangeslider_visible = True)
	fig.update_yaxes(title_text = 'Price')
	fig.update_layout(legend = dict(y = 0.5, traceorder = 'reversed', font_size = 16),
						autosize = False, height = 500, width = 800,
						margin=dict(l=2, r=2, b=20, t=30))
	st.plotly_chart(fig)
	


else: 
	st.warning('Wrong symbol, Use symbol finder tool to get the right one')

















