
import numpy as np
import pandas as pd


def _preprocess(df, drop_cols:list=None):
    columns=['datetime', 'expt_num', 'sitename', 'type', 'index', 'value']
    
    try:
        exp_num = df['expt_num'][0]
        sitename = df['sitename'][0]
        df.pop(['expt_num', 'sitename'], axis=1, inplace=True)

    except:
        print("Already dropped")
        
    
    # get unique sensor types
    sensor_types = pd.unique(df['type'])
    DATA_PER_SENSOR = sum(df['type'] == 'solution_pH')

    for type in sensor_types:
        mask = df['type'] == type
        print(f"{type}: {df[mask].shape[0] / DATA_PER_SENSOR:.2f} sensors")
        
    # construct sensor dict
    sensor_dict = {}
    for typ in sensor_types:
        _sensor = df.loc[df['type'] == typ] # select what type of sensor

        for ind in pd.unique(_sensor['index']): # select ith sensor
            sensor_dict[f'{typ}_{ind}'] = _sensor.loc[_sensor['index'] == ind]['value'].values
            #print(f"{typ}_{ind}: {_sensor.loc[_sensor['index'] == ind]['value'].isna().sum()} nan values")
            
    # construct df
    sensor_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in sensor_dict.items() ]))
    
    if drop_cols is not None:
        for col in drop_cols:
            sensor_df.pop(col) # drop column 
            print(f"{col} dropped")
    
    
    if True in sensor_df.isna().any().values: # check if there are NaN values
        sensor_df = sensor_df.interpolate(axis=0) # interpolate
        print("NaN values detected, interpolation applied")
    
    
    # average every instance of a sensor type
    temp_cols = [col for col in sensor_df.columns if "temperature" in col]
    humid_cols = [col for col in sensor_df.columns if "humidity" in col]
    li_cols = [col for col in sensor_df.columns if "light_intensity" in col]
    solution_EC = [col for col in sensor_df.columns if "solution_EC" in col]

    ph_cols = [col for col in sensor_df.columns if "solution_pH" in col]
    sm_cols = [col for col in sensor_df.columns if "soil_moisture" in col]

    ave_list = []

    temp_ave = sensor_df[temp_cols].mean(axis=1) # apply mean across column wise
    humid_ave = sensor_df[humid_cols].mean(axis=1)
    li_ave = sensor_df[li_cols].mean(axis=1)
    sm_ave = sensor_df[sm_cols].mean(axis=1)

    ave_list.append(temp_ave)
    ave_list.append(humid_ave)
    ave_list.append(li_ave)
    ave_list.append(sensor_df[solution_EC].squeeze())

    ave_list.append(sensor_df[ph_cols].squeeze())
    ave_list.append(sm_ave)

    ave_cols = ['temp_ave', 'humid_ave', 'li_ave', 'EC', 'ph', 'sm_ave']

    ave_dict = {}

    for k,v in zip(ave_cols, ave_list):
        ave_dict[k] = v


    ave_df = pd.DataFrame.from_dict(ave_dict)
    
    return ave_df
