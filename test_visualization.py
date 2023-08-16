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
    
    return df
    

def create_yearly_df(df):
    yearly_df = df.groupby(by = [df['Date'].dt.year, df['station']]).agg({
        'Total Polutan': 'mean',
        'PM2.5' : 'mean',
        'PM10' : 'mean',
        'SO2' : 'mean',
        'NO2' : 'mean',
        'CO' : 'mean',
        'O3' : 'mean'})    
    
    yearly_df.reset_index(inplace=True)
    
    yearly_df.rename(columns = {'Date':'year'}, inplace = True)
    
    return yearly_df

def create_hourly_df(df):
    hourly_df = df.groupby(by = [df['Hour'], df['station']]).agg({
        'Total Polutan': 'mean',
        'Temp' : 'mean',
        'O3' : 'mean'})    
    
    hourly_df.reset_index(inplace=True)
    
    return hourly_df

def mean_df(df):
   all_mean_df = df.groupby('station').mean()['Total Polutan'].reset_index().sort_values(['Total Polutan'], ascending = False)
   
   return all_mean_df

all_df = clean_data(df)

st.table(all_df.head())