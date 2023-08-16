import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import numpy as np
sns.set(style='dark')


df = pd.DataFrame()

for file in os.listdir(os.getcwd()):
    if file.endswith('.csv'):
        df = df.append(pd.read_csv(file))

def clean_data(df):
    df['Total Polutan'] = df['PM2.5'] + df['PM10'] + df['SO2'] + df['NO2'] + df['CO'] + df['O3']
    df = df.drop(['No', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM'], axis = 1)
    df['Date']=pd.to_datetime(df[['year','month','day']])
    df = df.drop(['year', 'month', 'day'], axis = 1)
    col = list(df.columns)
    df = df[[col[10]] + col[0:8] + [col[9]] + [col[8]]]
    df = df.rename(columns = {'hour':'Hour', 'TEMP':'Temp'})

all_df = clean_data(df)

st.table(all_df)