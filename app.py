#!/usr/bin/env python
# coding: utf-8

# # Import libraries:

# In[14]:

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st


# ## Models:

# In[73]:


# Get the absolute path of the current script
base_path = os.path.dirname(__file__)  # This is "/mount/src/focusing_web_or_app/model_dep"

# Construct correct paths to model and scalers
model_path = os.path.join(base_path, "model_lr.pkl")
scaler_path = os.path.join(base_path, "feature_scaler.pkl")
target_scaler_path = os.path.join(base_path, "target_scaler.pkl")

# Load the model and scalers
lr_model = pickle.load(open(model_path, "rb"))
feature_scaler = pickle.load(open(scaler_path, "rb"))
target_scaler = pickle.load(open(target_scaler_path, "rb"))


# # Deployment:

# In[83]:


## Creating a emptry dataframe for showing prediction data in a tabular format:
df = pd.DataFrame([], columns=['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Predicted Yearly Amount Spent(dollars)'])
Avg_ses_len = []
Time_on_app = []
Time_on_web = []
Len_of_mem = []
predicted_price = []

st.title("Yearly Amount Spent Prediction")
    
av_ses_len = st.sidebar.number_input('Enter Avg Session Length(in minutes) :', min_value=0.0, step=0.1)
time_on_app = st.sidebar.number_input('Enter Time on App(in minutes) :', min_value=0.0, step=0.1)
time_on_web = st.sidebar.number_input('Enter Time on Website(in minutes) :', min_value=0.0, step=0.1)
len_of_mem = st.sidebar.number_input('Enter Length of Membership(years) :', min_value=0.0, step=0.1)
user_data = [av_ses_len, time_on_app, time_on_web, len_of_mem]

# Append all user input:
Avg_ses_len.append(av_ses_len)
Time_on_app.append(time_on_app)
Time_on_web.append(time_on_web)
Len_of_mem.append(len_of_mem)

# scaling user input data:
scaled_usr_data = feature_scaler.transform([user_data])
predict = lr_model.predict(scaled_usr_data)
inverse_trans = target_scaler.inverse_transform([predict])[0][0]
predicted_price.append(inverse_trans)

if st.button('Predict'):
    df['Avg Session Length'] = Avg_ses_len
    df['Time on App'] = Time_on_app
    df['Time on Website'] = Time_on_web
    df['Length of Membership'] = Len_of_mem
    df['Predicted Yearly Amount Spent(dollars)'] = predicted_price
    st.subheader('Predictions:')
    st.table(df.style.format())


# In[ ]:




