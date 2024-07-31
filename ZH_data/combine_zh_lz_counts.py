##### Script to filter, clean, and process the traffic count data
## Inputs: 
# zurich_counts.h5
# luzern_jan_apr2015.csv 
## Output:
# luzern_and_zurich_counts.h5, luzern_counts.h5, zurich_counts.h5

import numpy as np
import pandas as pd

from utils_dataProcess import create_datetime_column, impute_erroneously_large_or_missing_measurement



counts_zurich = pd.read_hdf('zh_data/zurich_counts.h5')
counts_luzern = pd.read_csv('zh_data/luzern/luzern_jan_apr2015.csv')





counts_luzern = create_datetime_column(counts_luzern)
# check for possible duplicates for datetime and detid with differnet counts and filter them out by keeping only the first one
counts_luzern['unique_index'] = counts_luzern.groupby(['datetime', 'detid']).cumcount()
counts_luzern = counts_luzern[counts_luzern['unique_index'] == 0]

counts_luzern = counts_luzern.pivot(index='datetime', columns='detid', values='flow')

counts_luzern.index.name = None
counts_luzern.columns.name = None

# remove bad columns: 
#  over 10% of measurements are erroneously or missing
#  too much missing data (<100 counts total)
# not map matched
bad_sensors = ['ig11FD100_D4', 'ig11FD110_D11', 'ig11FD110_D28', 'ig11FD125_D4', 'ig11FD205_D87', 'ig11FD205_D90', 'ig11FD208_D2', 
               'ig11FD105_D7', 'ig11FD110_D33', 'ig11FD104_D43', 'ig11FD117_D9', 'ig11FD205_D92', 'ig11FD117_D12', 'ig11FD205_D89', 
               'ig11FD205_D91', 'ig11FD104_D42', 'ig11FD205_D88', 'ig11FD205_D2', 'ig11FD205_D81', 'ig11FD205_D84', 'ig11FD205_D85', 
               'ig11FD205_D85', 'ig11FD205_D86', 'ig11FD104_D5', 'ig11FD110_D13', 'ig11FD207_D6', 'ig11FD110_D30', 'ig11FD207_D5', 
               'ig11FD103_D6']
counts_luzern.drop(columns=bad_sensors, inplace=True)


counts_luzern = impute_erroneously_large_or_missing_measurement(counts_luzern)
counts_zurich = impute_erroneously_large_or_missing_measurement(counts_zurich)

counts_zurich.to_hdf('zh_data/zurich_counts.h5', key='df', mode='w')

counts_luzern.index = counts_luzern.index.astype(str)
counts_luzern.to_hdf('zh_data/luzern_counts.h5', key='df', mode='w')


counts_zurich.index = pd.to_datetime(counts_zurich.index)
counts_luzern.index = pd.to_datetime(counts_luzern.index)

luzern_and_zurich_counts = pd.concat([counts_zurich, counts_luzern], axis=1) #size (n row zurich + n row luzern)  x (n col zurich + n col luzern)

luzern_and_zurich_counts.to_hdf('zh_data/luzern_and_zurich_counts.h5', key='df', mode='w')
print("done")









