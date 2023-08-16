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
yearly_df = create_yearly_df(all_df)
hourly_df = create_hourly_df(all_df)
all_mean_df = mean_df(all_df)

min_year = yearly_df["year"].min()
max_year = yearly_df["year"].max()

min_hour = hourly_df['Hour'].min()
max_hour = hourly_df['Hour'].max()

with st.sidebar:
    st.image("beijing.jpg")
    
    selected_station = st.selectbox('Pilih Stasiun', all_df['station'].unique().tolist())
    

st.header('Beijing Station Air Quality ☁️')

st.subheader('Yearly Air Quality')

start_year, end_year = st.slider("Pilih Rentang Tahun", min_value=int(min_year), max_value=int(max_year), value=(int(min_year), int(max_year)))

yearly_pollution = []

for i in list(yearly_df['station'].unique()):
    yearly_pollution.append(i)
    
yearly_df = yearly_df[(yearly_df['year'] >= start_year) & (yearly_df['year'] <= end_year)]

fig, ax = plt.subplots(figsize=(18, 12))

ax.set_title('Average Total Air Pollution per Year on ' + selected_station + ' Station', 
             fontsize = 20, 
             pad=20, 
             fontweight = 'bold')

ax.plot(yearly_df.loc[yearly_df['station'] == selected_station]['year'], 
        yearly_df.loc[yearly_df['station'] == selected_station]['Total Polutan'], 
        color='blue',
        marker = 's',
        markersize = 20,
        linewidth = 5,
        markerfacecolor = 'blue',
        markeredgecolor = 'red')

ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

ax.set_xlabel('Year',size=20, labelpad = 20)
ax.set_ylabel('Total Polutan',size=20, labelpad = 20)

ax.set_xticks(list(yearly_df['year'].unique()))
 
st.pyplot(fig)