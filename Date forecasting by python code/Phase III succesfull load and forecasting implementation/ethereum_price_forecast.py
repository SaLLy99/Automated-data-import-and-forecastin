#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import regex as re
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

FILE_NAME = "./2023-06-01-13-18-37.csv"

eth = pd.read_csv(FILE_NAME)

print(f"{len(eth)} rows")

eth["Date"] = pd.to_datetime(eth['Date'])

last_date = eth["Date"].max()

print(f"Latest row is from {last_date}")

eth.head()


# In[2]:


eth.info()


# In[3]:


eth.describe()
eth.shape


# In[4]:


print('Total number of days :', eth.Date.nunique())
print('Total number of fields :', eth.shape[1])


# In[5]:


print("Null values :", eth.isnull().values.sum())
print("NA values :", eth.isna().values.any())


# In[6]:


print("Starting date :", eth.iloc[-1][0])
print("Ending date :", eth.iloc[0][0])
print("Duration :", eth.iloc[0][0]- eth.iloc[-1][0])


# In[7]:


monthwise = eth.groupby(pd.DatetimeIndex(eth.Date).month)[['Open']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reset_index()
monthwise['Date'] = new_order

monthwise


# In[8]:


fig = go.Figure()

fig.add_trace(go.Bar(
    x = monthwise.Date,
    y = monthwise['Open'],
    name = 'Stock Open Price',
    marker_color = 'black'
))
fig.update_layout(barmode = 'group', xaxis_tickangle = -45, 
                  title = 'Monthwise comparision for Open Prices')
fig.show()


# In[9]:


monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order


# In[10]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x = monthwise_high.Date,
    y = monthwise_high.High,
    name = 'Stock High Price',
    marker_color = 'purple'
))
fig.add_trace(go.Bar(
    x = monthwise_low.Date,
    y = monthwise_low.Low,
    name = 'Stock Low Price',
    marker_color='pink'
))

fig.update_layout(barmode='group', xaxis_tickangle = -45,
                  title=' Monthwise High and Low Price')
fig.show()


# In[11]:


names = cycle(['Eth Open Price','Eth High Price','Eth Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price analysis chart', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()


# In[12]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[13]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Stock'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[14]:


eth.plot(kind="line", x="Date", y="Open", figsize=(12,6))


# In[15]:


prophet_data = eth[["Date", "Open"]]

prophet_data = prophet_data.rename(columns = {
    "Date": "ds",
    "Open": "y"
})

prophet_data.head()


# In[16]:


pip install prophet


# In[17]:


from prophet import Prophet

prophet = Prophet(daily_seasonality=True)

prophet.fit(prophet_data)

print("Data fitted")


# In[18]:


future = prophet.make_future_dataframe(periods=365, include_history=False)

future.tail()


# In[19]:


forecast = prophet.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[20]:


forecast.info()


# In[21]:


import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(dpi=100)

fig.set_facecolor("white")

prophet_plot_forecast_fig = prophet.plot(forecast, ax=fig.gca());

prophet_plot_forecast_fig.savefig('forecast_details.png')


# In[22]:


prophet.plot_components(forecast);


# In[23]:


PLOT_COLUMS = [
    "Price",
    "Price (forecast)",
]

mpl.style.use("seaborn")

result_df = prophet_data.copy()
print(result_df.tail(1).rename(columns = {"y": "yhat"}))

# Add first result from forecast as y to connect dots
result_df = result_df.append(result_df.tail(1).rename(columns = {"y": "yhat"}))

result_df = result_df.append(forecast)

result_df = result_df.rename(columns = {
    "ds": "Date",
    "y": "Price",
    "yhat": "Price (forecast)"
})

fig = plt.figure(dpi=100)

fig.set_facecolor("white")

plot = result_df.plot(x="Date", y=PLOT_COLUMS, figsize=(15, 8), ax=fig.gca())

plot_fig = plot.get_figure()

plot_fig.savefig('forecast.png')

