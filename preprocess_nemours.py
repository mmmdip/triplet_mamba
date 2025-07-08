import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.preprocessing import OrdinalEncoder

RAW_DATA_PATH = './datasets/nemours'

def get_demographics( data ):
    val = data.fact.split('|')
    gender, ethinicity, race, zipcode = val   
    age = calculate_age( pd.to_datetime( data.val ).date(), data.visit_start_datetime )  
    return pd.Series([ gender, ethinicity, race, zipcode[:-4], age ], index = [ 'gender', 'ethnicity', 'race', 'zipcode', 'age' ])                

def calculate_age( dob, ref_date ):
    return round( ( ref_date - dob ).days / 365 )

def read_ts( raw_data_path ):
    facts = facts = pd.read_csv( raw_data_path + '/pt_facts.csv' )
    facts['visit_start_datetime'] = pd.to_datetime( facts.visit_start_datetime ).dt.date
    # facts = facts[:100000]
    
    demo = facts.loc[facts.fact_typ == 'DEMO'].set_index('deid').drop_duplicates()
    meas = facts.loc[facts.fact_typ == 'MEAS']
    
    wts = meas.loc[meas.fact_id.isin( wt_codes )]
    meas = meas.loc[meas.fact_id.isin( meas_codes )]
    
    # Drop missing/inconsistent values
    meas['val'] = pd.to_numeric(meas['val'], errors='coerce')
    meas = meas[meas['val'].notna()]
    
    del facts
    
    ts = []
    oc = []
    # begin = time.time()
    for deid, d in tqdm( meas.groupby( "deid" )):
        w = wts.loc[wts.deid == deid]
        dem = demo.loc[deid]
        if w.empty or dem.empty:
            continue
        
        dem_date = dem.visit_start_datetime
        dem = get_demographics( dem )
        demographics = pd.DataFrame( dem ).reset_index().rename( columns = { 'index' : 'fact', 0 : 'val' })
        demographics = demographics.assign( visit_start_datetime = dem_date, deid = deid, timestamp = 0 )
        ts.append( demographics )
        
        d = d[['visit_start_datetime', 'fact', 'val']]
        # d.loc['timestamp'] = d.visit_start_datetime.apply( lambda x : ( x - dem_date ).days )
        d = d.assign(
            deid = deid,
            timestamp = d.visit_start_datetime.copy().apply( lambda x : ( x - dem_date ).days )
            )
        ts.append( d )
        
        out = {
            'deid' : deid,
            'start_weight' : w.loc[w.fact_id == weight_code,'val'].values[0],
            'nadir_weight' : w.loc[w.fact_id == weight_code,'val'].min(),
            'end_weight' : w.loc[w.fact_id == weight_code,'val'].values[-1],
            # 'start_BMIp95' : w.loc[w.fact_id == BMIp95_code,'val'].values[0],
            # 'nadir_BMIp95' : w.loc[w.fact_id == BMIp95_code,'val'].min(),
            # 'end_BMIp95' : w.loc[w.fact_id == BMIp95_code,'val'].values[-1]
            }
        oc.append( list(out.values()) )
        # break
    ts = pd.concat( ts )
    # oc = pd.DataFrame( oc, columns = [ 'deid', 'start_weight', 'nadir_weight', 'end_weight', 'start_BMIp95', 'nadir_BMIp95', 'end_BMIp95']).set_index( 'deid' ).astype('float32')    
    oc = pd.DataFrame( oc, columns = [ 'deid', 'start_weight', 'nadir_weight', 'end_weight']).set_index( 'deid' ).astype('float32')    
    oc = oc.assign( 
        wt_chng_pct = ( oc.start_weight - oc.end_weight ) / oc.start_weight * 100,
        wt_chng_pct_max = ( oc.start_weight - oc.nadir_weight ) / oc.start_weight * 100,
        # bmi_chng_pct = ( oc.start_BMIp95 - oc.end_BMIp95 ) / oc.start_BMIp95 * 100,
        # bmi_chng_pct_max = ( oc.start_BMIp95 - oc.nadir_BMIp95 ) / oc.start_BMIp95 * 100,
        ).astype('float32')
    # finish = time.time()
    # elapsed_time = finish - begin
    # print('\nElapsed time: {}'.format(time.strftime('%H:%M:%S', time.gmtime( elapsed_time ))))

    ts.drop( columns = ['visit_start_datetime'], inplace = True )
    ts.rename(columns={'timestamp': 'minute', 'fact': 'variable',
                       'val': 'value', 'deid': 'ts_id'}, inplace = True)
    oc[ "out_w_t" ] = 0
    oc[ "out_w_m" ] = 0
    # oc[ "out_b_t" ] = 0
    # oc[ "out_b_m" ] = 0

    oc.loc[ oc.wt_chng_pct >= 5.0, 'out_w_t' ] = 1
    oc.loc[ oc.wt_chng_pct_max >= 5.0, 'out_w_m' ] = 1
    # oc.loc[ oc.bmi_chng_pct >= 5.0, 'out_b_t' ] = 1
    # oc.loc[ oc.bmi_chng_pct_max >= 5.0, 'out_b_m' ] = 1
    
    oc = oc.reset_index()
    oc.rename(columns = {'deid': 'ts_id'}, inplace = True)
    
    return ts, oc


# measurements filters
# 3027114 : 'Cholesterol [Mass/volume] in Serum or Plasma'
# 3007070 : 'Cholesterol in HDL [Mass/volume] in Serum or Plasma'
# 3007352 : 'Cholesterol in VLDL [Mass/volume] in Serum or Plasma'
# 3044491 : 'Cholesterol non HDL [Mass/volume] in Serum or Plasma'
# 3012888 : 'Diastolic blood pressure'
# 3004501 : 'Glucose [Mass/volume] in Serum or Plasma'
# 3027018 : 'Heart rate'
# 3004410 : 'Hemoglobin A1c/Hemoglobin.total in Blood'
# 3024171 : 'Respiratory rate'
# 3004249 : 'Systolic blood pressure'
# 3020891 : 'Body temperature'

# 40762638 : 'Body mass index (BMI) [Percentile] Per age and sex'
# 3013762 : 'Body weight Measured'

meas_codes = [ 3027114, 3007070, 3007352, 3044491, 3012888, 3004501, 3027018, 3004410, 3024171, 3004249, 3020891 ]
wt_codes = [ 40762638, 3013762 ]
BMIp95_code = 40762638
weight_code = 3013762

begin = time.time()
ts, oc = read_ts( RAW_DATA_PATH )
finish = time.time()
elapsed_time = finish - begin
print('\nElapsed time: {}'.format(time.strftime('%H:%M:%S', time.gmtime( elapsed_time ))))

ts_ids = sorted(list(ts.ts_id.unique()))
oc = oc.loc[oc.ts_id.isin(ts_ids)]

# Drop duplicates.
ts = ts.drop_duplicates()

# Categorical encoding
cat_vars = ['gender', 'ethnicity', 'race']
for c, d in ts.loc[ts.variable.isin( cat_vars )].groupby( "variable" ):
    enc = OrdinalEncoder( dtype = np.int8 )
    ts.loc[ts.variable == c, 'value'] = enc.fit_transform( d[['value']] )
ts['value'] = ts['value'].astype('float64')

# Generate split.
train_valid_ids = list(oc.ts_id)
np.random.seed(2196)
np.random.shuffle(train_valid_ids)
train_p = int(0.6 * len(train_valid_ids))
val_p = int(0.8 * len(train_valid_ids))
train_ids = train_valid_ids[:train_p]
valid_ids = train_valid_ids[train_p:val_p]
test_ids = train_valid_ids[val_p:]

# Store data.
os.makedirs('../data/processed', exist_ok=True)
pickle.dump([ts, oc, train_ids, valid_ids, test_ids],
            open('../data/processed/nemours.pkl', 'wb'))
