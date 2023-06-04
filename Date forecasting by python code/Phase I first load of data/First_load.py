#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name # set the region of the instance

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")


# In[2]:


bucket_name = 'raw-sample-file' 
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# In[3]:


try:
  urllib.request.urlretrieve ("https://raw-sample-file.s3.us-west-2.amazonaws.com/ethereum_price.csv", "ethereum_price.csv)
  print('Success: downloaded ethereum_price.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./ethereum_price.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[4]:


try:
  urllib.request.urlretrieve ("https://raw-sample-file.s3.us-west-2.amazonaws.com/ethereum_price.csv", "ethereum_price.csv")
  print('Success: downloaded ethereum_price.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./ethereum_price.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[5]:


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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU

import warnings
warnings.filterwarnings('ignore')



# In[6]:


eth = pd.read_csv('./ethereum_price.csv')


# In[7]:


eth.head()


# In[8]:


eth['Date'] = pd.to_datetime(eth.Date)

for i in range(len(eth)):
    eth['Price'][i] = float(re.sub(',', '', eth['Price'][i]))
    eth['Open'][i] = float(re.sub(',', '', eth['Open'][i]))
    eth['High'][i] = float(re.sub(',', '', eth['High'][i]))
    eth['Low'][i] = float(re.sub(',', '', eth['Low'][i]))
    eth['Change %'][i] = float(re.sub('%', '', eth['Change %'][i]))
    if eth['Vol.'][i][-1] == 'K':
        eth['Vol.'][i] = int(float(re.sub('K', '', eth['Vol.'][i])) * 1000) 
    elif eth['Vol.'][i][-1] == 'M':
        eth['Vol.'][i] = int(float(re.sub('M', '', eth['Vol.'][i])) * 1000000) 

eth.head()


# In[9]:


eth.shape


# In[10]:


print('Total number of days :', eth.Date.nunique())
print('Total number of fields :', eth.shape[1])


# In[11]:


print("Null values :", eth.isnull().values.sum())
print("NA values :", eth.isna().values.any())


# In[12]:


print("Starting date :", eth.iloc[-1][0])
print("Ending date :", eth.iloc[0][0])
print("Duration :", eth.iloc[0][0]- eth.iloc[-1][0])


# In[13]:


monthwise = eth.groupby(pd.DatetimeIndex(eth.Date).month)[['Open']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reset_index()
monthwise['Date'] = new_order

monthwise


# In[14]:


fig = go.Figure()

fig.add_trace(go.Bar(
    x = monthwise.Date,
    y = monthwise['Open'],
    name = 'Stock Open Price',
    marker_color = 'pink'
))
fig.update_layout(barmode = 'group', xaxis_tickangle = -45, 
                  title = 'Monthwise comparision for Open Prices')
fig.show()


# In[15]:


monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order


# In[16]:


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


# In[17]:


names = cycle(['Eth Open Price','Eth High Price','Eth Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price analysis chart', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()


# In[18]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[19]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Stock'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[20]:


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])


# In[21]:


fig = px.line(open_stock, x = open_stock.Date, y = open_stock.Open, labels = {'Date':'Date','Open':'Open Stock Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict Ethereum close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[22]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[23]:


train_size = int(len(open_stock)*0.75)
test_size = len(open_stock) - train_size
train_data , test_data = open_stock[0:train_size, :] ,open_stock[train_size:len(open_stock),:1]
print("Train_data :", train_data.shape)
print("Test_data :", test_data.shape)


# In[24]:


def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[25]:


time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test", y_test.shape)


# In[26]:


#Reshaping input to be of format [samples, time steps, features] which is reuqired for LSTM
x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape)


# In[27]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[28]:


model.summary()


# In[29]:


history = model.fit(x_train_lstm, y_train, validation_data = (x_test_lstm, y_test), epochs = 200, batch_size = 32, verbose = 1)


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[31]:


pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[32]:


pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[33]:


train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)


# In[34]:


boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[35]:


bucket_name='raw-sample-file'
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[36]:


import boto3
s3 = boto3.resource('s3')
for key in bucket.objects.all():
  print 's3://{}/{}.format(bucket,key.key)


# In[37]:


import boto3
s3 = boto3.resource('s3')
for key in bucket.objects.all():
  print 's3://{}.format(bucket,key.key)


# In[38]:


import boto3
s3 = boto3.resource('s3')
for key in bucket.objects.all():
  print key


# In[39]:


import boto3
s3 = boto3.resource('s3')
for key in bucket.objects.all():
  print(key)


# In[40]:


import boto3
s3 = boto3.resource('s3')
for my_bucket_object in s3.objects.all():
    print(my_bucket_object)


# In[41]:


import boto3
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('raw-sample-file')

for file in my_bucket.objects.all():
    print(file.key)


# In[42]:


bucket_name='raw-sample-file'
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3:/train'.format(bucket_name, prefix), content_type='csv')


# In[43]:


bucket_name='raw-sample-file'
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}'.format(bucket_name, prefix), content_type='csv')


# In[44]:


train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)


# In[45]:


bucket_name='raw-sample-file'
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}'.format(bucket_name, prefix), content_type='csv')


# In[ ]:


bucket_name='raw-sample-file'
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/ethereum_price.csv')).upload_file('ethereum_price.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}'.format(bucket_name, prefix), content_type='csv')


# In[46]:


sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)


# In[47]:


xgb.fit({'train': s3_input_train})


# In[48]:


sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)


# In[49]:


s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[50]:


xgb.fit({'train': s3_input_train})


# In[51]:


print(f"{len(eth)} rows")

eth["Date"] = pd.to_datetime(eth['Date'])

last_date = eth["Date"].max()

print(f"Latest row is from {last_date}")

eth.head()


# In[52]:


eth.info()


# In[53]:


eth.describe()


# In[54]:


eth.plot(kind="line", x="Date", y="Close", figsize=(12,6))



# In[55]:


eth.plot(kind="line", x="Date", y="Close", figsize=(12,6))


# In[56]:


prophet_data = eth[["Date", "Close"]]

prophet_data = prophet_data.rename(columns = {
    "Date": "ds",
    "Close": "y"
})

prophet_data.head()



# In[57]:


from prophet import Prophet

prophet = Prophet(daily_seasonality=True)

prophet.fit(prophet_data)

print("Data fitted")



# In[58]:


import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(dpi=100)

fig.set_facecolor("white")

prophet_plot_forecast_fig = prophet.plot(forecast, ax=fig.gca());

prophet_plot_forecast_fig.savefig('forecast_details.png')



# In[ ]:




