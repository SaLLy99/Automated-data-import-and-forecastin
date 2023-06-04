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


# In[1]:


try:
  urllib.request.urlretrieve ("https://raw-sample-file.s3.us-west-2.amazonaws.com/processed_data/2023-06-01-13-18-37.csv", "2023-06-01-13-18-37.csv")
  print('Success: downloaded 2023-06-01-13-18-37.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./2023-06-01-13-18-37.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[2]:


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


# In[3]:


try:
  urllib.request.urlretrieve ("https://raw-sample-file.s3.us-west-2.amazonaws.com/processed_data/2023-06-01-13-18-37.csv", "2023-06-01-13-18-37.csv")
  print('Success: downloaded 2023-06-01-13-18-37.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./2023-06-01-13-18-37.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[4]:


train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)


# In[5]:


pd.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[6]:


model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[7]:


model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format("raw-sample-data", prefix), content_type='csv')


# In[8]:


model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-sample-data"/"processed_data"/train'.format("raw-sample-data", prefix), content_type='csv')


# In[9]:


model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-sample-data"/"processed_data"/train'.format("raw-sample-data", prefix), content_type='csv')


# In[ ]:


boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-sample-data"/"processed_data"/train'.format("raw-sample-data", prefix), content_type='csv')


# In[10]:


model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-sample-data"/"processed_data"/train'.format("raw-sample-data", prefix), content_type='csv')


# In[11]:


boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-sample-data"/"processed_data"/train'.format("raw-sample-data", prefix), content_type='csv')


# In[12]:


boto3.Session().resource('s3').Bucket("raw-sample-data").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-sample-data"/"processed_data"/train'.format("raw-sample-data", prefix), content_type='csv')


# In[14]:


bucket_name = 'model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket("raw-processed-file").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-processed-file"/"processed_data"/train'.format("raw-processed-file", prefix), content_type='csv')' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-weast-2':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# In[15]:


model_data.to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket("raw-processed-file").Object(os.path.join(prefix, './train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://"raw-processed-file"/"processed_data"/train'.format("raw-processed-file", prefix), content_type='csv')


# In[16]:


sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)


# In[17]:


xgb.fit({'train': s3_input_train})


# In[18]:


df_total = pd.read_csv("./2023-06-01-13-18-37.csv")
df_total = df_total.drop("Unnamed: 0", 1)
df_total


# In[19]:


df_total = pd.read_csv("./2023-06-01-13-18-37.csv")
df_total


# In[20]:


import matplotlib.pyplot as plt

df_total["Price"].plot()
plt.show()


# In[21]:


import tensorflow as tf


# In[22]:


conda install jupyter notebook


# In[ ]:


conda update -n base -c conda-forge conda


# In[ ]:


jupyter notebook


# In[ ]:


import tensorflow as tf


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

series = df_total[[column for column in df_total.columns if column not in ["Date(UTC)"]]]
series = series.applymap(lambda value: value.replace(",", "") if type(value) is str else value)
series = series.to_numpy()
scaler = MinMaxScaler()
series = scaler.fit_transform(series)


# In[1]:


import tensorflow as tf


# In[ ]:


conda deactivate 


# In[ ]:


pip install tensorflow 


# In[ ]:


conda activate second_load


# In[ ]:


jupyter notebook


# In[ ]:


import tensorflow as tf


# In[1]:


conda install -c conda-forge tensorflow


# In[ ]:


conda update -n base -c conda-forge conda


# In[1]:


import tensorflow as ft


# In[2]:


pip install tensorflow


# In[3]:


rom sklearn.preprocessing import MinMaxScaler
import numpy as np

series = df_total[[column for column in df_total.columns if column not in ["Date(UTC)"]]]
series = series.applymap(lambda value: value.replace(",", "") if type(value) is str else value)
series = series.to_numpy()
scaler = MinMaxScaler()
series = scaler.fit_transform(series)
     


# In[4]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

series = df_total[[column for column in df_total.columns if column not in ["Date(UTC)"]]]
series = series.applymap(lambda value: value.replace(",", "") if type(value) is str else value)
series = series.to_numpy()
scaler = MinMaxScaler()
series = scaler.fit_transform(series)


# In[5]:


import matplotlib.pyplot as plt

df_total["price"].plot()
plt.show()
     


# In[6]:


from functools import reduce

df_total = reduce(lambda df1, df2: pd.merge(df1, df2, on='Date(UTC)'), df_list)
df_total = df_total.rename(columns={"Value (Wei)": "avg gas price"})
df_total
     


# In[7]:


eth.head()


# In[8]:


eth = pd.read_csv('./2023-06-01-13-18-37.csv')


# In[9]:


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


# In[10]:


eth = pd.read_csv('./2023-06-01-13-18-37.csv')


# In[11]:


eth.head()


# In[12]:


eth['Date'] = pd.to_datetime(eth.Date)


# In[13]:


eth.shape


# In[14]:


print('Total number of days :', eth.Date.nunique())
print('Total number of fields :', eth.shape[1])


# In[15]:


print("Null values :", eth.isnull().values.sum())
print("NA values :", eth.isna().values.any())


# In[16]:


monthwise = eth.groupby(pd.DatetimeIndex(eth.Date).month)[['Open']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reset_index()
monthwise['Date'] = new_order

monthwise


# In[17]:


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


# In[18]:


monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order


# In[19]:


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


# In[20]:


names = cycle(['Eth Open Price','Eth High Price','Eth Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price analysis chart', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()


# In[21]:


names = cycle(['Etherium Open Price','Etherium High Price','Etherium Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price  variation during 2017-2023', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()


# In[22]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[23]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Time'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close & price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'yellow')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[24]:


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])


# In[25]:


fig = px.line(open_stock, x = open_stock.Date, y = open_stock.Open, labels = {'Date':'Date','Open':'Open Stock Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict Ethereum close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[26]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[27]:


train_size = int(len(open_stock)*0.75)
test_size = len(open_stock) - train_size
train_data , test_data = open_stock[0:train_size, :] ,open_stock[train_size:len(open_stock),:1]
print("Train_data :", train_data.shape)
print("Test_data :", test_data.shape)


# In[28]:


def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[29]:


time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test", y_test.shape)


# In[30]:


x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape)


# In[31]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[32]:


get_ipython().system('pip install tensorflow==2.2-rc3')


# In[33]:


pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.2-rc3-py3-none-any.whl


# In[1]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[2]:


import tensorflow as tf


# In[3]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[4]:


from keras.models import Sequential


# In[5]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[6]:


from tensorflow.keras.layers import LSTM, GRU


# In[7]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[8]:


time_step = 15


# In[9]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[10]:


tf.keras.backend.set_image_data_format("channels_last")


# In[11]:


import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip tensorflow numpy scikit-learn pandas')


# In[12]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[13]:


from keras.layers import Activation, Dense, Dropout


# In[14]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[15]:


model.summary()


# In[16]:


history = model.fit(x_train_lstm, y_train, validation_data = (x_test_lstm, y_test), epochs = 200, batch_size = 32, verbose = 1)


# In[17]:


x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape


# In[18]:


x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape)


# In[19]:


time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test", y_test.shape)


# In[20]:


train_size = int(len(open_stock)*0.75)
test_size = len(open_stock) - train_size
train_data , test_data = open_stock[0:train_size, :] ,open_stock[train_size:len(open_stock),:1]
print("Train_data :", train_data.shape)
print("Test_data :", test_data.shape)


# In[21]:


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])


# In[22]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[23]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Time'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close & price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'yellow')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[ ]:


import plotly.express as px


# In[ ]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Time'})

fig.update_traces(marker_line_width = 2, opacity = 0.8)

fig.update_layout(title_text = 'Stock close & price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'yellow')

fig.update_xaxes(showgrid = False)

fig.update_yaxes(showgrid = False)

fig.show()

-----------------


# In[26]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[27]:


eth = pd.read_csv('./2023-06-01-13-18-37.csv')


# In[28]:


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


# In[29]:


import pandas as pd


# In[30]:


eth = pd.read_csv('./2023-06-01-13-18-37.csv')


# In[31]:


eth.head()


# In[32]:


eth['Date'] = pd.to_datetime(eth.Date)


# In[33]:


names = cycle(['Etherium Open Price','Etherium High Price','Etherium Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price  variation during 2017-2023', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()


# In[34]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[35]:


monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order


# In[36]:


print("Null values :", eth.isnull().values.sum())
print("NA values :", eth.isna().values.any())




monthwise = eth.groupby(pd.DatetimeIndex(eth.Date).month)[['Open']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reset_index()
monthwise['Date'] = new_order

monthwise


# In[37]:


monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order


# In[38]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Time'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close & price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'yellow')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[39]:


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])


# In[40]:


fig = px.line(open_stock, x = open_stock.Date, y = open_stock.Open, labels = {'Date':'Date','Open':'Open Stock Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict Ethereum close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[41]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[42]:


from sklearn.preprocessing import MinMaxScaler


# In[43]:


pip install -U numpy


# In[44]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[45]:


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
from tensorflow.keras.layers import LSTM


# In[46]:


pip install -U numpy


# In[47]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[1]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[2]:


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




eth = pd.read_csv('./2023-06-01-13-18-37.csv')


eth.head()


eth['Date'] = pd.to_datetime(eth.Date)



print('Total number of days :', eth.Date.nunique())
print('Total number of fields :', eth.shape[1])





print("Null values :", eth.isnull().values.sum())
print("NA values :", eth.isna().values.any())




monthwise = eth.groupby(pd.DatetimeIndex(eth.Date).month)[['Open']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reset_index()
monthwise['Date'] = new_order

monthwise



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






names = cycle(['Etherium Open Price','Etherium High Price','Etherium Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price  variation during 2017-2023', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()




open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()




monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order






fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Time'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close & price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'yellow')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])




fig = px.line(open_stock, x = open_stock.Date, y = open_stock.Open, labels = {'Date':'Date','Open':'Open Stock Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict Ethereum close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()




del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)




#check training data

train_size = int(len(open_stock)*0.75)
test_size = len(open_stock) - train_size
train_data , test_data = open_stock[0:train_size, :] ,open_stock[train_size:len(open_stock),:1]
print("Train_data :", train_data.shape)
print("Test_data :", test_data.shape)



def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
	
	
	
	
	
	
	
time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test", y_test.shape)



#start training



x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape


#dealing with GRU related issue 
#from keras.models import Sequential
#import tensorflow as tf

#from tensorflow.keras.layers import LSTM, GRU

#tf.keras.backend.set_image_data_format("channels_last")

#import sys
#!{sys.executable} -m pip install --upgrade pip tensorflow numpy scikit-learn pandas



tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[3]:


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




eth = pd.read_csv('./2023-06-01-13-18-37.csv')


eth.head()


eth['Date'] = pd.to_datetime(eth.Date)



print('Total number of days :', eth.Date.nunique())
print('Total number of fields :', eth.shape[1])





print("Null values :", eth.isnull().values.sum())
print("NA values :", eth.isna().values.any())




monthwise = eth.groupby(pd.DatetimeIndex(eth.Date).month)[['Open']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reset_index()
monthwise['Date'] = new_order

monthwise



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






names = cycle(['Etherium Open Price','Etherium High Price','Etherium Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price  variation during 2017-2023', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()




open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()




monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order






fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Time'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close & price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'yellow')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])




fig = px.line(open_stock, x = open_stock.Date, y = open_stock.Open, labels = {'Date':'Date','Open':'Open Stock Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict Ethereum close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()




del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)




#check training data

train_size = int(len(open_stock)*0.75)
test_size = len(open_stock) - train_size
train_data , test_data = open_stock[0:train_size, :] ,open_stock[train_size:len(open_stock),:1]
print("Train_data :", train_data.shape)
print("Test_data :", test_data.shape)



def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
	
	
	
	
	
	
	
time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test", y_test.shape)



#start training



x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape)


#dealing with GRU related issue 
#from keras.models import Sequential
#import tensorflow as tf

#from tensorflow.keras.layers import LSTM, GRU

#tf.keras.backend.set_image_data_format("channels_last")

#import sys
#!{sys.executable} -m pip install --upgrade pip tensorflow numpy scikit-learn pandas



tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[4]:


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


# In[5]:


monthwise_high = eth.groupby(pd.DatetimeIndex(eth.Date).month)['High'].max()
monthwise_high = monthwise_high.reset_index()
monthwise_high['Date'] = new_order

monthwise_low = eth.groupby(pd.DatetimeIndex(eth.Date).month)['Low'].min()
monthwise_low = monthwise_low.reset_index()
monthwise_low['Date'] = new_order


# In[6]:


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


# In[7]:


names = cycle(['Eth Open Price','Eth High Price','Eth Low Price'])

fig = px.line(eth, x = eth.Date, y = [eth['Open'], eth['High'], eth['Low']],
             labels = {'date': 'Date','value':'Eth value'})
fig.update_layout(title_text = 'Ethereum Price analysis chart', font_size = 15, font_color = 'black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)

fig.show()


# In[8]:


open_eth = eth[['Date', 'Open']]
print(open_eth.shape)
open_eth.head()


# In[9]:


fig = px.line(open_eth, x = open_eth.Date, y = open_eth.Open,labels = {'date':'Date','close':'Close Stock'})
fig.update_traces(marker_line_width = 2, opacity = 0.8)
fig.update_layout(title_text = 'Stock close price chart', plot_bgcolor = 'white', font_size = 15, font_color = 'black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[10]:


open_eth = open_eth[open_eth['Date'] > '2022-03-08']
open_stock = open_eth.copy()
print("Total data for prediction: ",open_stock.shape[0])


# In[11]:


fig = px.line(open_stock, x = open_stock.Date, y = open_stock.Open, labels = {'Date':'Date','Open':'Open Stock Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict Ethereum close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()


# In[12]:


del open_stock['Date']
scaler = MinMaxScaler(feature_range = (0,1))
open_stock = scaler.fit_transform(np.array(open_stock).reshape(-1,1))
print(open_stock.shape)


# In[13]:


train_size = int(len(open_stock)*0.75)
test_size = len(open_stock) - train_size
train_data , test_data = open_stock[0:train_size, :] ,open_stock[train_size:len(open_stock),:1]
print("Train_data :", train_data.shape)
print("Test_data :", test_data.shape)


# In[14]:


def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]    
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[15]:


time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test", y_test.shape)


# In[ ]:


x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape)


# In[16]:


x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train_lstm.shape, x_test_lstm.shape)


# In[17]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[18]:


tf.keras.backend.set_image_data_format("channels_last")


# In[19]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[20]:


model.summary()


# In[21]:


history = model.fit(x_train_lstm, y_train, validation_data = (x_test_lstm, y_test), epochs = 200, batch_size = 32, verbose = 1)


# In[22]:


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
print(train_predict.shape, test_predict.shape)


# In[23]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 


# In[24]:


print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))


# In[25]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict)


# In[26]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))


# In[27]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))


# In[28]:


print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


# In[29]:


look_back = time_step
train_predict_plot = np.empty_like(open_stock)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back : len(train_predict) + look_back, :] = train_predict
print("Train predicted data: ", train_predict_plot.shape)

# shift test predictions for plotting
test_predict_plot = np.empty_like(open_stock)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(open_stock) - 1, :] = test_predict
print("Test predicted data: ", test_predict_plot.shape)

names = cycle(['Original Open price','Train predicted Open price','Test predicted Open price'])

plotdf = pd.DataFrame({'Date': open_eth['Date'],
                       'original_open': open_eth['Open'],
                      'train_predicted_open': train_predict_plot.reshape(1,-1)[0].tolist(),
                      'test_predicted_open': test_predict_plot.reshape(1,-1)[0].tolist()})
plotdf['original_open'] = plotdf['original_open'].astype(np.float64)

fig = px.line(plotdf, x = plotdf['Date'], y = [plotdf['original_open'], plotdf['train_predicted_open'], plotdf['test_predicted_open']],
              labels = {'value':'Ethereum price','Date': 'Date'})
fig.update_layout(title_text = 'Comparision between original Open price vs predicted Open price',
                  plot_bgcolor = 'white', font_size = 15, font_color = 'black', legend_title_text = 'Open Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[30]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 45
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[31]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[32]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[33]:


from numpy import array


# In[34]:


lst_output=[]
n_steps=time_step
i=0
pred_days = 30


# In[35]:


while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[36]:


print(test_data)


# In[37]:


print(x_input)


# In[38]:


print(temp_input)


# In[39]:


lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[40]:


model.predict(x_input, verbose=0)


# In[41]:


look_back = time_step
train_predict_plot = np.empty_like(open_stock)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back : len(train_predict) + look_back, :] = train_predict
print("Train predicted data: ", train_predict_plot.shape)

# shift test predictions for plotting
test_predict_plot = np.empty_like(open_stock)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(open_stock) - 1, :] = test_predict
print("Test predicted data: ", test_predict_plot.shape)

names = cycle(['Original Open price','Train predicted Open price','Test predicted Open price'])

plotdf = pd.DataFrame({'Date': open_eth['Date'],
                       'original_open': open_eth['Open'],
                      'train_predicted_open': train_predict_plot.reshape(1,-1)[0].tolist(),
                      'test_predicted_open': test_predict_plot.reshape(1,-1)[0].tolist()})
plotdf['original_open'] = plotdf['original_open'].astype(np.float64)

fig = px.line(plotdf, x = plotdf['Date'], y = [plotdf['original_open'], plotdf['train_predicted_open'], plotdf['test_predicted_open']],
              labels = {'value':'Ethereum price','Date': 'Date'})
fig.update_layout(title_text = 'Comparision between original Open price vs predicted Open price',
                  plot_bgcolor = 'white', font_size = 15, font_color = 'black', legend_title_text = 'Open Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[42]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[43]:


model.summary()


# In[44]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[45]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[46]:


from numpy import array


# In[47]:


lst_output=[]
n_steps=time_step
i=0
pred_days = 30


# In[48]:


while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[49]:


yhat = model.predict(x_input, verbose=0)
print("{} day output {}".format(i,yhat))


# In[50]:


yhat = model.predict(x_input, verbose=0)


# In[51]:


print(x_input)


# In[52]:


print(modal)


# In[53]:


print(model)


# In[54]:


model.history()


# In[55]:


model.sammary()


# In[56]:


model.summary()


# In[57]:


lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(
    x_input, batch_size=None, verbose=0, steps=None, callbacks=None
)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(
    x_input, batch_size=None, verbose=0, steps=None, callbacks=None
)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[58]:


future = prophet.make_future_dataframe(periods=365, include_history=False)

future.tail()


# In[59]:


from tensorflow.keras.models import Sequential


# In[60]:


tf.keras.backend.clear_session()
model = Sequential()
model.add(GRU(32, return_sequences = True, input_shape = (time_step, 1)))
model.add(GRU(32, return_sequences = True))
model.add(GRU(32))
model.add(Dropout(0.20))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[61]:


model.summary()


# In[62]:


history = model.fit(x_train_lstm, y_train, validation_data = (x_test_lstm, y_test), epochs = 200, batch_size = 32, verbose = 1)


# In[63]:


history = model.fit(x_train_lstm, y_train, validation_data = (x_test_lstm, y_test), epochs = 200, batch_size = 32, verbose = 1)


# In[64]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[65]:


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
print(train_predict.shape, test_predict.shape)


# In[66]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 


# In[67]:


print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))


# In[68]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))


# In[69]:


print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))


# In[70]:


print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


# In[71]:


look_back = time_step
train_predict_plot = np.empty_like(open_stock)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back : len(train_predict) + look_back, :] = train_predict
print("Train predicted data: ", train_predict_plot.shape)

# shift test predictions for plotting
test_predict_plot = np.empty_like(open_stock)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(open_stock) - 1, :] = test_predict
print("Test predicted data: ", test_predict_plot.shape)

names = cycle(['Original Open price','Train predicted Open price','Test predicted Open price'])

plotdf = pd.DataFrame({'Date': open_eth['Date'],
                       'original_open': open_eth['Open'],
                      'train_predicted_open': train_predict_plot.reshape(1,-1)[0].tolist(),
                      'test_predicted_open': test_predict_plot.reshape(1,-1)[0].tolist()})
plotdf['original_open'] = plotdf['original_open'].astype(np.float64)

fig = px.line(plotdf, x = plotdf['Date'], y = [plotdf['original_open'], plotdf['train_predicted_open'], plotdf['test_predicted_open']],
              labels = {'value':'Ethereum price','Date': 'Date'})
fig.update_layout(title_text = 'Comparision between original Open price vs predicted Open price',
                  plot_bgcolor = 'white', font_size = 15, font_color = 'black', legend_title_text = 'Open Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[72]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[73]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = self.model.predic(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        #x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[74]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predic(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        #x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[75]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        #x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[76]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[77]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0,batch_size=50)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0,batch_size=50)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[78]:


timesteps = input_shape[0] if self.time_major else input_shape[1]

    TypeError: Exception encountered when calling layer 'gru' (type GRU).
    
    'NoneType' object is not subscriptable


# In[79]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        model.build(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
		 model.build(x_input)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


model.predict(np.random.randint(1, 5, size=(50, 230, 230, 1)), batch_size=50)


# In[80]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        model.build(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        model.build(x_input)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


model.predict(np.random.randint(1, 5, size=(50, 230, 230, 1)), batch_size=50)


# In[81]:


print(model.layers)


# In[82]:


model 


# In[83]:


print(model.weights)


# In[84]:


model.preduct(input_data)


# In[85]:


model.predict(input_data)


# In[86]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        model.build(x_input)
        yhat = model.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        model.build(x_input)
        yhat = model.predict(x_input)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[ ]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        model.build(x_input)
        
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        model.build(x_input)
        #yhat = model.predict(x_input, verbose=0)
        
               
print("Output of predicted next days: ", len(lst_output))


model.predict(np.random.randint(1, 5, size=(50, 230, 230, 1)), batch_size=50)


# In[ ]:


# x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        model.build(x_input)
        
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        model.build(x_input)
        #yhat = model.predict(x_input, verbose=0)


# In[ ]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = self.model.predic(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[ ]:


print(model)


# In[88]:


print("Output of predicted next days: ")


# In[89]:


pd.set_option('display.max_columns', 50000) 


# In[90]:


pd.set_option('display.max_columns', 50)


# In[91]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[92]:


look_back = time_step
train_predict_plot = np.empty_like(open_stock)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back : len(train_predict) + look_back, :] = train_predict
print("Train predicted data: ", train_predict_plot.shape)

# shift test predictions for plotting
test_predict_plot = np.empty_like(open_stock)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(open_stock) - 1, :] = test_predict
print("Test predicted data: ", test_predict_plot.shape)

names = cycle(['Original Open price','Train predicted Open price','Test predicted Open price'])

plotdf = pd.DataFrame({'Date': open_eth['Date'],
                       'original_open': open_eth['Open'],
                      'train_predicted_open': train_predict_plot.reshape(1,-1)[0].tolist(),
                      'test_predicted_open': test_predict_plot.reshape(1,-1)[0].tolist()})
plotdf['original_open'] = plotdf['original_open'].astype(np.float64)

fig = px.line(plotdf, x = plotdf['Date'], y = [plotdf['original_open'], plotdf['train_predicted_open'], plotdf['test_predicted_open']],
              labels = {'value':'Ethereum price','Date': 'Date'})
fig.update_layout(title_text = 'Comparision between original Open price vs predicted Open price',
                  plot_bgcolor = 'white', font_size = 15, font_color = 'black', legend_title_text = 'Open Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[93]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)


# In[94]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(open_stock[len(open_stock)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days Open price','Predicted next 30 days Open price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Ethereum price','index': 'Timestamp'})
fig.update_layout(title_text='Comparing last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[95]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]
​
last_original_days_value = temp_mat
next_predicted_days_value = temp_mat
​
last_original_days_value[0:time_step+1] = scaler.inverse_transform(open_stock[len(open_stock)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
​
new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})
​
names = cycle(['Last 15 days Open price','Predicted next 30 days Open price'])
​
fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Ethereum price','index': 'Timestamp'})
fig.update_layout(title_text='Comparing last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
​
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[96]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(open_stock[len(open_stock)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days Open price','Predicted next 30 days Open price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Ethereum price','index': 'Timestamp'})
fig.update_layout(title_text='Comparing last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[97]:


lstmdf=open_stock.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[98]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)


# In[99]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(open_stock[len(open_stock)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days Open price','Predicted next 30 days Open price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Ethereum price','index': 'Timestamp'})
fig.update_layout(title_text='Comparing last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[100]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[101]:


print(train_predict)


# In[102]:


print(test_predict)


# In[103]:


print(train_predict.shape, test_predict.shape)


# In[104]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 


# In[105]:


print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))


# In[106]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))


# In[107]:


print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))


# In[108]:


print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


# In[ ]:




