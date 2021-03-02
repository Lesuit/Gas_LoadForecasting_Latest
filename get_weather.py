# -*- coding: utf-8 -*-
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import csv
import pickle
# # Conversion between pd.Timestamp and Python datetime!!!
# # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html

# time_now = datetime.now()
# #convert datetime to UTC-timestamp
# UTC_timestamp = time_now.timestamp()

# # convert datetime to pd.Timestamp
# pdTimeStamp = pd.Timestamp(time_now)

# # conver datetime.timestamp to pd.Timestamp
# pdTimeStamp = pd.Timestamp(UTC_timestamp, unit='s', tz='US/Mountain')
# # complete list of timezones: https://stackoverflow.com/questions/13866926/is-there-a-list-of-pytz-timezones

# # convert pd.Timestamp to datetime
# pdTimeStamp.to_pydatetime()

### Below are a searies of functions that use weather APIs to get historical and forecast weather data.

def get_pd_timestamps_from_now(NumOfLookbackDays, NumOfForecastDays):
    time_now = datetime.now()
    UTC_timestamp = time_now.timestamp()
    pd_timestamps = []
    for i in range(-NumOfLookbackDays, NumOfForecastDays+1, 1):
        cur_timestamp = pd.Timestamp(UTC_timestamp + i*3600*24, unit='s', tz='US/Mountain')
        pd_timestamps.append(cur_timestamp)
    return [pd.Timestamp(year=x.year, month=x.month, day=x.day) for x in pd_timestamps]


def get_hist_avg_weather(NumOfLookbackDays=21, NumOfForecastDays=14): 
    with open('hist_avg_weather_df.pkl', 'rb') as input:
        hist_mean_weather_df = pickle.load(input)
    pd_timestamps = get_pd_timestamps_from_now(NumOfLookbackDays, NumOfForecastDays)
    dayofyear_list = [x.dayofyear for x in pd_timestamps]
    output_hist_avgs = hist_mean_weather_df.loc[dayofyear_list][['Cgy_avg_temperature', 
                                                                'Edm_avg_temperature', 
                                                                'Cgy_avg_wind_speed',
                                                                'Edm_avg_wind_speed']]

    output_hist_avgs.columns = ['CGY HIST Temp (Celsius)', 'EDM HIST Temp (Celsius)', 
                                'CGY HIST Windspeed (mph)', 'EDM HIST Windspeed (mph)']
    output_hist_avgs['date'] = pd_timestamps
    return output_hist_avgs


def get_past_actual_weather(cityId, NumOfLookbackDays=22, NumOfForecastDays=0):   # 22, -1
    pd_timestamps = get_pd_timestamps_from_now(NumOfLookbackDays, NumOfForecastDays)
    StartDate = pd_timestamps[0].strftime('%m/%d/%Y')
    EndDate = pd_timestamps[-1].strftime('%m/%d/%Y')
    d_df = get_historical_daily_weather(StartDate, EndDate, cityId)
    h_df = get_historical_hourly_weather(StartDate, EndDate, cityId)   # When you query during evening, the hourly data for last                                                                        # night might not be avaiable yet.
    df = pd.merge(d_df, h_df, on='date')
    columnnames = ['date', 'Min Temp (Celsius)', 'Max Temp (Celsius)', 
                   'Avg Temp (Celsius)', 'Avg Hourly Temp (Celsius)', 
                   'Avg Windspeed (mph)']
    df.columns = columnnames
    return df


def get_forecast_weather(cityId):
    df_DailyMinMax = get_forecast_daily_MinMax_temp(cityId)
    df_DailyAverTemp = get_forecast_daily_average_temp(cityId)
    df_hourly = get_forecast_hourly_weather(cityId)
    df = pd.merge(df_DailyMinMax, df_DailyAverTemp, on='date')
    df = pd.merge(df, df_hourly, on='date')
    columnnames = ['date', 'Min Temp (Celsius)', 'Max Temp (Celsius)', 
                   'Avg Temp (Celsius)', 'Avg Hourly Temp (Celsius)', 
                   'Avg Windspeed (mph)']
    df.columns = columnnames
    return df


def get_forecast_hourly_weather(cityId):   # CYYC=Calgary.  CYEG=Edmonton
    temp_url = "https://www.wsitrader.com/Account/Login?ReturnUrl=/Services/CSVDownloadService.svc/GetHourlyForecast?" \
               "region=NA%26SiteId={}%26TempUnits=C".format(cityId)

    r = requests.post(temp_url, data={"Account": 'atco', "Password":'weather1', 
                     'Profile': 'Kui.Pan@atco.com', 'Action': 'Enter'}) 
    decoded_content = r.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    df = pd.DataFrame(my_list[1:])
    df.columns = df.iloc[0,:].values
    df = df.iloc[1:,:]
    df['Temp'] = df[' Temp'].apply(lambda x: float(x))
    df['Wind'] = df[' WindSpeed(mph)'].apply(lambda x: float(x))
    df['LocalTime'] = df['LocalTime'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
    df = df[['LocalTime', 'Temp', 'Wind']]
    dates, aver_temp, aver_wind = [], [], []
    for i in range(15):
        lower_bound, upper_bound = i*24, (i+1)*24
        dates.append(df['LocalTime'].values[lower_bound])
        aver_temp.append(np.mean(df['Temp'].values[lower_bound:upper_bound]))
        aver_wind.append(np.mean(df['Wind'].values[lower_bound:upper_bound]))
    dates = pd.to_datetime(dates)
    df = pd.DataFrame()
    df['date'] = dates
    df['hourly_avg_temperature'] = aver_temp
    df['hourly_avg_wind_speed'] = aver_wind
    return df

def get_forecast_daily_average_temp(cityId):    
    temp_url = "https://www.wsitrader.com/Account/Login?ReturnUrl=/Services/" \
    "CSVDownloadService.svc/GetCityTableForecast?" \
    "IsCustom=false%26CurrentTabName=AverageTemp%26TempUnits=C%26Id=AESO%26Region=NA" 

    r = requests.post(temp_url, data={"Account": 'atco', "Password":'weather1', 
                     'Profile': 'Kui.Pan@atco.com', 'Action': 'Enter'}) 
    decoded_content = r.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    df = pd.DataFrame(my_list)
    # We find which row we need to keep based on the cityId
    row_num = -1
    for i, city_name in enumerate(df.iloc[:,0].values):
        if cityId in str(city_name):
            row_num = i
            break        
    dates = [datetime.strptime(x, '%m/%d/%Y') for x in my_list[1][1:16]]
    AverTemp = [float(x) for x in my_list[row_num][1:16]]
    output_df = pd.DataFrame()
    output_df['date'] = dates
    output_df['avg_temp'] = AverTemp
    return output_df


def get_forecast_daily_MinMax_temp(cityId):
    temp_url = "https://www.wsitrader.com/Account/Login?ReturnUrl=/Services/" \
               "CSVDownloadService.svc/GetCityTableForecast?" \
               "IsCustom=false%26CurrentTabName=MinMax%26TempUnits=C%26Id=AESO%26Region=NA" 
    r = requests.post(temp_url, data={"Account": 'atco', "Password":'weather1', 
                     'Profile': 'Kui.Pan@atco.com', 'Action': 'Enter'}) 
    decoded_content = r.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    df = pd.DataFrame(my_list)
    # We find which row we need to keep based on the cityId
    row_num = -1
    for i, city_name in enumerate(df.iloc[:,0].values):
        if cityId in str(city_name):
            row_num = i
            break
    dates_double = [datetime.strptime(x, '%m/%d/%Y') for x in my_list[1][1:-2]]
    MinMax = my_list[row_num][1:-2]
    dates, MinTemp, MaxTemp = [], [], []
    for i in range(15):
        dates.append(dates_double[i*2])
        MinTemp.append(MinMax[i*2])
        MaxTemp.append(MinMax[i*2+1])
    MinTemp = [float(x) for x in MinTemp]
    MaxTemp = [float(x) for x in MaxTemp]
    output_df = pd.DataFrame()
    output_df['date'] = dates
    output_df['min_temp'] = MinTemp
    output_df['max_temp'] = MaxTemp
    return output_df


def get_historical_daily_weather(StartDate, EndDate, CityId):  #01/01/2021, 02/20/2021, "CYYC"/"CYEG"
    new_url = "https://www.wsitrader.com/Account/Login?ReturnUrl=/Services/" \
    "CSVDownloadService.svc/GetHistoricalObservations?" \
    "%26HistoricalProductID=HISTORICAL_DAILY_OBSERVED%26StartDate={}%26EndDate={}" \
    "%26IsDisplayDates=false%26IsTemp=true%26TempUnits=C%26CityIds[]={}".format(StartDate, EndDate, CityId)
    r = requests.post(new_url, data={"Account": 'atco', "Password":'weather1', 
                     'Profile': 'Kui.Pan@atco.com', 'Action': 'Enter'}) 
    decoded_content = r.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    df = pd.DataFrame(my_list)
    df = df.iloc[3:,:-1]
    df.columns = ['date', 'min_temp', 'max_temp', 'avg_temperature']
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%b-%Y'))
    for i in range(1, 4):
        col_name = df.columns[i]
        df[col_name] = df[col_name].apply(lambda x: float(x))
    return df


def get_historical_hourly_weather(StartDate, EndDate, CityId):  #01/01/2021, 02/20/2021, "CYYC"/"CYEG"
    new_url = "https://www.wsitrader.com/Account/Login?ReturnUrl=/Services/" \
    "CSVDownloadService.svc/GetHistoricalObservations?" \
    "%26HistoricalProductID=HISTORICAL_HOURLY_OBSERVED%26DataTypes[]=windSpeed" \
    "%26DataTypes[]=temperature%26TempUnits=C" \
    "%26StartDate={}%26EndDate={}%26CityIds[]={}".format(StartDate, EndDate, CityId)
    r = requests.post(new_url, data={"Account": 'atco', "Password":'weather1', 
                     'Profile': 'Kui.Pan@atco.com', 'Action': 'Enter'}) 
    decoded_content = r.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    df = pd.DataFrame(my_list)
    df = df.iloc[2:,]
    df.columns = ['date', 'hour', 'temp', 'wind']
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    for i in range(1, 4):
        col_name = df.columns[i]
        df[col_name] = df[col_name].apply(lambda x: float(x))

    dates, aver_temp, aver_wind = [], [], []
    for i in range(df.shape[0]//24):
        lower_bound, upper_bound = i*24, (i+1)*24
        dates.append(df['date'].values[lower_bound])
        aver_temp.append(np.mean(df['temp'].values[lower_bound:upper_bound]))
        aver_wind.append(np.mean(df['wind'].values[lower_bound:upper_bound]))
    dates = pd.to_datetime(dates)
    final_df = pd.DataFrame()
    final_df['date'] = dates
    final_df['hourly_avg_temperature'] = aver_temp
    final_df['hourly_avg_wind_speed'] = aver_wind
    return final_df


def get_historical_avg_weather_for_same_day():
    edm_daily = get_historical_daily_weather('01/01/2017', '01/01/2021', 'CYEG')
    edm_hourly = get_historical_hourly_weather('01/01/2017', '01/01/2021', 'CYEG')
    cgy_daily = get_historical_daily_weather('01/01/2017', '01/01/2021', 'CYYC')
    cgy_hourly = get_historical_hourly_weather('01/01/2017', '01/01/2021', 'CYYC')
    edm_weather_df = pd.merge(edm_daily, edm_hourly, on='date')
    cgy_weather_df = pd.merge(cgy_daily, cgy_hourly, on='date')
    weather_df = pd.merge(edm_weather_df, cgy_weather_df, on='date')

    weather_df.columns = ['date', 'Edm_min_temp', 'Edm_max_temp', 'Edm_avg_temperature', 
                          'Edm_hourly_avg_temperature',  'Edm_avg_wind_speed',
                          'Cgy_min_temp', 'Cgy_max_temp', 'Cgy_avg_temperature', 
                          'Cgy_hourly_avg_temperature',  'Cgy_avg_wind_speed']

    final_list = []
    for i in range(1, 367):
        temp_df = weather_df[weather_df.date.dt.dayofyear==i]
        means = temp_df.describe().iloc[1,:].to_list()
        final_list.append(means)
    hist_mean_weather_df = pd.DataFrame(final_list, columns=['Edm_min_temp', 'Edm_max_temp', 'Edm_avg_temperature',
                                                             'Edm_hourly_avg_temperature',  'Edm_avg_wind_speed',
                                                             'Cgy_min_temp', 'Cgy_max_temp', 'Cgy_avg_temperature', 
                                                             'Cgy_hourly_avg_temperature',  'Cgy_avg_wind_speed'])
    hist_mean_weather_df.index = [i for i in range(1, 367)]

    with open('hist_avg_weather_df.pkl', 'wb') as output:
        pickle.dump(hist_mean_weather_df, output, pickle.HIGHEST_PROTOCOL)
    return hist_mean_weather_df