#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:08:32 2021

@author: kuipan
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from google.cloud import storage
from get_weather import get_hist_avg_weather
from get_weather import get_past_actual_weather
from get_weather import get_forecast_weather
import holidays
import re

zone_profiles = ['101APT', '101COMM', '101INDU', '101LCOM', '101MAPT', '101MCOM', '101RES',
                 '2601COM_LO', '2601RES', '2601RUR', 
                 '103APT', '103COMM', '103LCOM', '103MAPT', '103MCOM', '103MIND','103MRES','103RES']

city_names = ['CYEG', 'CYYC']
zone_to_city = dict()
zone_to_city['101'] = ['CYEG'] #Edmonton
zone_to_city['2601'] = ['CYEG'] #Edmonton
zone_to_city['103'] = ['CYYC'] #Calgary

# Upload files to storage bucket
def upload_cloud_storage(filepath):
    destination_blob_name = filepath.split('/')[-1]
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('short_term_gas_forecast')
    blog = bucket.blob(destination_blob_name)
    blog.upload_from_filename(filepath)

# Download files from storage bucket
def download_cloud_storage(filepath):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('short_term_gas_forecast')
    blob1 = bucket.blob(filepath)
    blob1.download_to_filename('/tmp/' + filepath)
    

def filter_df_by_date(df, dates, col_name):
    new_df = df[df[col_name].isin(dates)]
    return new_df

def LoadTrainedModels_helper(zone_profile, Models_Dict, scaler_Dict, ymax_Dict, cus_type):
    for i in range(3):
        saved_model_name = 'trained_models/' + zone_profile + '_' + cus_type + str(i+1) + '.h5'
        model = keras.models.load_model(saved_model_name)
        Models_Dict[saved_model_name] = model
    
    saved_scaler_name = 'trained_models/' + zone_profile + '_' + cus_type + 'scaler.pkl'
    scaler_Dict[saved_scaler_name] = pickle.load(open(saved_scaler_name, 'rb'))
    
    saved_ymax_name = 'trained_models/' + zone_profile + '_' + cus_type + 'ymax.pkl'
    ymax_Dict[saved_ymax_name] = pickle.load(open(saved_ymax_name, 'rb'))

def LoadTrainedModels():
    Models_Dict, scaler_Dict, ymax_Dict = dict(), dict(), dict()
    for zone_profile in zone_profiles:
        for cus_type in ['RES', 'SMB']:
            LoadTrainedModels_helper(zone_profile, Models_Dict, scaler_Dict, ymax_Dict, cus_type)
    return [Models_Dict, scaler_Dict, ymax_Dict]


def ProcessWeatherForMLInputs(PastAndForecastedWeatherPerCity):
    ca_holidays = holidays.Canada(state='AB' ,years=list(i for i in range(2016, 2035)))
    PastAndForecastedWeatherPerZone = dict()
    for key in zone_to_city.keys():
        city_list = zone_to_city[key]
        if len(city_list)==1:
            temp_df = PastAndForecastedWeatherPerCity[city_list[0]]
            temp_df.columns = ['date', 'min_temp', 'max_temp', 'avg_temp', 
                               'avg_hourly_temp', 'avg_wind_speed']
            final_df = temp_df
        else:
            city1, city2 = city_list
            temp1 = PastAndForecastedWeatherPerCity[city1]
            temp2 = PastAndForecastedWeatherPerCity[city2]
            merge_df = pd.merge(temp1, temp2, on='date')
            merge_df.columns = ['date', 
                                'min_temp1', 'max_temp1', 'avg_temp1', 'avg_hourly_temp1', 'avg_wind_speed1',
                                'min_temp2', 'max_temp2', 'avg_temp2', 'avg_hourly_temp2', 'avg_wind_speed2']   
            temp_df = pd.DataFrame()
            temp_df['date'] = merge_df['date']
            for colname in ['min_temp', 'max_temp', 'avg_temp', 'avg_hourly_temp', 'avg_wind_speed']:
                values1 = merge_df[colname+'1'].values
                values2 = merge_df[colname+'2'].values
                temp_df[colname] = [np.mean([values1[i], values2[i]]) for i in range(len(values1))]
            final_df = temp_df
    
        final_df = final_df.assign(**{'HDD New': 
                                      [max(0, 18-x) for x in final_df['avg_temp'].values]})

        final_df = final_df.assign(**{'Dayofweek': [x.dayofweek for x in final_df['date']], 
                                      'Monthofyear': [x.month for x in final_df['date']]})

        final_df = final_df.assign(**{'2D_Dayofweek': 
                                      [np.sin(2*np.pi*(x)/7) for x in final_df['Dayofweek']],
                                      '2D_Monthofyear': 
                                      [np.sin(2*np.pi*x/12) for x in final_df['Monthofyear']],
                                      'Weekend': 
                                      ((final_df['Dayofweek'] == 5) | (final_df['Dayofweek'] == 6)).astype(float),
                                      'temp*wind':
                                      final_df['avg_temp']*final_df['avg_wind_speed']})

        final_df = final_df.assign(**{'holiday': [1 if x in ca_holidays else 0 for x in final_df['date']]})

        final_df = final_df.assign(**{
            'temp_squared': final_df['avg_temp']**2,
            'temp_cubic': final_df['avg_temp']**3})

        final_df = final_df.assign(**{
            'temp_diff': final_df['avg_temp'].diff()})

        final_df = final_df.assign(**{
            'wind_squared': final_df['avg_wind_speed']**2,
            'wind_cubic': final_df['avg_wind_speed']**3
        })
        final_df.dropna(inplace=True)
        PastAndForecastedWeatherPerZone[key] = final_df
    return PastAndForecastedWeatherPerZone

# Add pandas columns values as a new column.
def add_column_values(input_df):
    final_sum = 0
    for col_name in input_df.columns:
        final_sum += input_df[col_name]
    return list(final_sum)

# Calculate site counts weighted average temperature
def CalculateSiteCountsWeightedAvgTemp(cgy_cnt, edm_cnt, cgy_temp, edm_temp):
    weighted_avg = []
    for a, b, c, d in zip(cgy_cnt, edm_cnt, cgy_temp, edm_temp):
        weighted_avg.append((a*c+b*d)/(a+b))
    return weighted_avg


def model_prediction(zone_profile, PastAndForecastedWeatherPerZone,
                     Models_Dict, scaler_Dict, ymax_Dict, output_df, cus_type):
    NumberZone = re.sub('\D', '', zone_profile)
    df = PastAndForecastedWeatherPerZone[NumberZone]
    x_raw = df.iloc[:,1:].values
        
    saved_scaler_name = 'trained_models/' + zone_profile + '_' + cus_type + 'scaler.pkl'
    scaler = scaler_Dict[saved_scaler_name]
    
    saved_ymax_name = 'trained_models/' + zone_profile + '_' + cus_type + 'ymax.pkl'
    ymax = ymax_Dict[saved_ymax_name]

    x = scaler.transform(x_raw)
    y_preds = []
    for i in range(3):
        saved_model_name = 'trained_models/' + zone_profile + '_' + cus_type + str(i+1) + '.h5'
        model = Models_Dict[saved_model_name]
        y_pred = list(model.predict(x)*ymax)
        y_preds.append(y_pred)
    mean_ypred = [np.mean([y_preds[0][i], y_preds[1][i], y_preds[2][i]]) for i in range(len(y_preds[0]))]
    output_df[zone_profile + '_' + cus_type] = mean_ypred


def ProcessForecastByCustomerType(site_counts, output_df, cus_type):
    ## Generate total consumptions for each zone profile
    consumptions = []
    col_names = []
    for x in site_counts.columns[1:]:
        col_name = x.split('.')[0]
        try:
            counts = site_counts[x].values
            avg_consum = output_df[col_name + '_' + cus_type].values
            total_consum = np.multiply(counts, avg_consum)
            consumptions.append(total_consum)
            col_names.append(x)
        except:
            continue   
    df = pd.DataFrame(np.array(consumptions).transpose())
    df.columns = col_names

    ## Find the zone profile col names
    fixed_cols = []
    float_cols = []
    fixed_cgy = []
    float_cgy = []
    fixed_edm = []
    float_edm = []
    for col in df.columns:
        if col.endswith('1'):
            float_cols.append(col)
            if col.startswith('103'):
                float_cgy.append(col)
            else:
                float_edm.append(col)
        else:
            fixed_cols.append(col)
            if col.startswith('103'):
                fixed_cgy.append(col)
            else:
                fixed_edm.append(col)        

    # Generate output columns
    total_forecast_GJ = add_column_values(df)
    fixed_forecast_GJ = add_column_values(df[fixed_cols])
    float_forecast_GJ = add_column_values(df[float_cols])

    total_site_count_Cgy = add_column_values(site_counts[fixed_cgy + float_cgy])
    fixed_site_count_Cgy = add_column_values(site_counts[fixed_cgy])
    float_site_count_Cgy = add_column_values(site_counts[float_cgy])

    total_site_count_Edm = add_column_values(site_counts[fixed_edm + float_edm])
    fixed_site_count_Edm = add_column_values(site_counts[fixed_edm])
    float_site_count_Edm = add_column_values(site_counts[float_edm])
    
    return [total_forecast_GJ, fixed_forecast_GJ, float_forecast_GJ,
            total_site_count_Cgy, fixed_site_count_Cgy, float_site_count_Cgy, 
            total_site_count_Edm, fixed_site_count_Edm, float_site_count_Edm]


def GetLoadForecastingResult(Models_Dict, scaler_Dict, ymax_Dict, filename):  # 'uploads/NG ST Forecast Report.xlsm'
    PastAndForecastedWeatherPerCity = dict()
    for city in city_names:
        past = get_past_actual_weather(city, 22, 0) # We can modify the backcast 22 days here.
        forecast = get_forecast_weather(city)
        PastAndForecastedWeatherPerCity[city] = pd.concat([past, forecast], axis=0)
    final_df = pd.merge(PastAndForecastedWeatherPerCity['CYYC'], 
                        PastAndForecastedWeatherPerCity['CYEG'], on='date')
    final_df.columns = ['date', 'CGY MIN Temp (Celsius)', 'CGY MAX Temp (Celsius)', 'CGY AVG Temp (Celsius)',
                       'CGY Hourly AVG Temp (Celsius)', 'CGY AVG Windspeed (mph)',
                       'EDM MIN Temp (Celsius)', 'EDM MAX Temp (Celsius)', 'EDM AVG Temp (Celsius)',
                       'EDM Hourly AVG Temp (Celsius)', 'EDM AVG Windspeed (mph)']
    
    PastAndForecastedWeatherPerZone = ProcessWeatherForMLInputs(PastAndForecastedWeatherPerCity)

    output_df = pd.DataFrame()
    output_df['date'] = PastAndForecastedWeatherPerZone['101']['date']   # For electricity forecasting, change '101'
                                                                         # to 1001
    for zone_profile in zone_profiles:
        for cus_type in ['RES', 'SMB']:
            model_prediction(zone_profile, PastAndForecastedWeatherPerZone, 
                             Models_Dict, scaler_Dict, ymax_Dict, output_df, cus_type)

    download_cloud_storage(filename)
    print ('successfully downloaed site counts file from blogstorage!!!', '\n')
    site_counts = pd.read_excel('/tmp/' + filename, sheet_name=None)
    site_counts_RES = site_counts['RES Forecast']
    site_counts_SMB = site_counts['SMB Forecast']
    site_counts_RES['Date'] = [pd.Timestamp(year=x.year, month=x.month, day=x.day) 
                               for x in site_counts_RES['Date']]
    site_counts_SMB['Date'] = [pd.Timestamp(year=x.year, month=x.month, day=x.day) 
                               for x in site_counts_SMB['Date']]    
    
    common_dates = []
    for d in site_counts_RES['Date']:
        if d in output_df['date'].values:
            common_dates.append(d)
    print (common_dates)
    final_df = filter_df_by_date(final_df, common_dates, 'date')
    output_df = filter_df_by_date(output_df, common_dates, 'date')
    site_counts_RES = filter_df_by_date(site_counts_RES, common_dates, 'Date')
    site_counts_SMB = filter_df_by_date(site_counts_SMB, common_dates, 'Date')
           
#     [total_forecast_GJ, fixed_forecast_GJ, float_forecast_GJ,
#      total_site_count_Cgy, fixed_site_count_Cgy, float_site_count_Cgy, 
#      total_site_count_Edm, fixed_site_count_Edm, float_site_count_Edm]
    RES_results = ProcessForecastByCustomerType(site_counts_RES, output_df, 'RES')
    SMB_results = ProcessForecastByCustomerType(site_counts_SMB, output_df, 'SMB')
    
    total_GJ = list(np.array(RES_results[0]) + np.array(SMB_results[0]))
    RES_Fixed_GJ = RES_results[1]
    RES_Floating_GJ = RES_results[2]
    SMB_Fixed_GJ = SMB_results[1]
    SMB_Floating_GJ = SMB_results[2]

    total_site_count_Cgy = list(np.array(RES_results[3]) + np.array(SMB_results[3]))
    RES_Fixed_Cgy = RES_results[4]
    RES_Floating_Cgy = RES_results[5]
    SMB_Fixed_Cgy = SMB_results[4]
    SMB_Floating_Cgy = SMB_results[5]

    total_site_count_Edm = list(np.array(RES_results[6]) + np.array(SMB_results[6]))
    RES_Fixed_Edm = RES_results[7]
    RES_Floating_Edm = RES_results[8]
    SMB_Fixed_Edm = SMB_results[7]
    SMB_Floating_Edm = SMB_results[8]
           
    #Determine how many days for backcast and forecast
    time_now = datetime.now()
    UTC_timestamp = time_now.timestamp()
    cur_timestamp = pd.Timestamp(UTC_timestamp, unit='s', tz='US/Mountain')
    cur_timestamp = pd.Timestamp(year=cur_timestamp.year, month=cur_timestamp.month, day=cur_timestamp.day)
    
    NumOfBackcastDays = len([x for x in common_dates if x<cur_timestamp])
    NumOfForecastDays = len([x for x in common_dates if x>=cur_timestamp])
    Final_Output_File = pd.DataFrame()
    Final_Output_File['date'] = output_df['date']

    Final_Output_File['Total Actual GJ'] = ['NA']*(NumOfBackcastDays + NumOfForecastDays)
    
    Final_Output_File['Total Forecast GJ'] = ['NA']*NumOfBackcastDays + total_GJ[NumOfBackcastDays:]
    Final_Output_File['RES Fixed Forecast GJ'] = ['NA']*NumOfBackcastDays + RES_Fixed_GJ[NumOfBackcastDays:]
    Final_Output_File['RES Floating Forecast GJ'] = ['NA']*NumOfBackcastDays + RES_Floating_GJ[NumOfBackcastDays:]
    Final_Output_File['SMB Fixed Forecast GJ'] = ['NA']*NumOfBackcastDays + SMB_Fixed_GJ[NumOfBackcastDays:]
    Final_Output_File['SMB Floating Forecast GJ'] = ['NA']*NumOfBackcastDays + SMB_Floating_GJ[NumOfBackcastDays:]
    
    
    Final_Output_File['Total Backcast GJ'] = total_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays
    Final_Output_File['RES Fixed Backcast GJ'] = RES_Fixed_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays
    Final_Output_File['RES Floating Backcast GJ'] = RES_Floating_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays
    Final_Output_File['SMB Fixed Backcast GJ'] = SMB_Fixed_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays
    Final_Output_File['SMB Floating Backcast GJ'] = SMB_Floating_GJ[:NumOfBackcastDays] + ['NA']*NumOfForecastDays

    
    
    Final_Output_File['Total Site Count CGY'] = total_site_count_Cgy
    Final_Output_File['RES Fixed Site Count CGY'] = RES_Fixed_Cgy
    Final_Output_File['RES Floating Site Count CGY'] = RES_Floating_Cgy
    Final_Output_File['SMB Fixed Site Count CGY'] = SMB_Fixed_Cgy
    Final_Output_File['SMB Floating Site Count CGY'] = SMB_Floating_Cgy
    

    Final_Output_File['Total Site Count EDM'] = total_site_count_Edm
    Final_Output_File['RES Fixed Site Count EDM'] = RES_Fixed_Edm
    Final_Output_File['RES Floating Site Count EDM'] = RES_Floating_Edm
    Final_Output_File['SMB Fixed Site Count EDM'] = SMB_Fixed_Edm
    Final_Output_File['SMB Floating Site Count EDM'] = SMB_Floating_Edm

    # Handling weather output columns
    temp_df = final_df
    temp_df['Site Count Weighted Average Temp (Celsius)'] = CalculateSiteCountsWeightedAvgTemp(total_site_count_Cgy, 
                                                                                               total_site_count_Edm, 
                                                                                               temp_df['CGY AVG Temp (Celsius)'].tolist(),
                                                                                               temp_df['EDM AVG Temp (Celsius)'].tolist())

    # Get historial average weather on the same day as today.
    HIST_AVG = get_hist_avg_weather()
    Final_Output_File = pd.merge(Final_Output_File, temp_df, on='date')
    Final_Output_File = pd.merge(Final_Output_File, HIST_AVG, on='date')

    cur_time = cur_timestamp.strftime('%m-%d-%Y')
    report_filename = cur_time + '_NG_ST_Forecast&Backcast.xlsx'
    ReportFilePath = '/tmp/' + report_filename
    Final_Output_File.to_excel(ReportFilePath)
    
    upload_cloud_storage(ReportFilePath)
    
    return report_filename