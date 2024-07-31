import pandas as pd

# function to convert the day and interval columns to a datetime column
def create_datetime_column(df): 
    # get the hour and minute from the interval column
    df['minutes'] = (df['interval']-179) / 60
    df['hours'] = (df['minutes'] // 60).astype(int)
    df['remaining_minutes'] = (df['minutes'] % 60).astype(int)

    # create a datetime column
    df['datetime_str'] = df['day'] + ' ' + df['hours'].astype(str).str.zfill(2) + ':' + df['remaining_minutes'].astype(str).str.zfill(2)
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M')

    # drop intermediate columns
    df.drop(columns=['datetime_str', 'hours', 'remaining_minutes', 'minutes'], inplace=True)

    return df

def impute_erroneously_large_or_missing_measurement(df, threshold=2000):
    df['datetime'] = df.index
    df.index = range(len(df))

    
    for column in df.columns:
        if column == 'datetime':
            continue
        indices_to_impute = (df[(df[column] > threshold) | df[column].isnull()].index).tolist()

        if len(indices_to_impute) > 2000:
            print(column)
            continue


        for index in indices_to_impute:
            if index == 0:
                # If the first value is erroneous, take the next non-erroneous value
                next_valid_index = index + 1
                while next_valid_index in indices_to_impute and next_valid_index < len(df):
                    next_valid_index += 1
                df.loc[index, column] = df.loc[next_valid_index, column]
            elif index == len(df) - 1:
                # If the last value is erroneous, take the previous non-erroneous value
                previous_valid_index = index - 1
                while previous_valid_index in indices_to_impute and previous_valid_index >= 0:
                    previous_valid_index -= 1
                df.loc[index, column] = df.loc[previous_valid_index, column]
            else:
                # For other values, average the nearest non-erroneous previous and next values
                previous_valid_index = index - 1
                next_valid_index = index + 1
                while previous_valid_index in indices_to_impute and previous_valid_index >= 0:
                    previous_valid_index -= 1
                while next_valid_index in indices_to_impute and next_valid_index < len(df):
                    next_valid_index += 1
                df.loc[index, column] = (df.loc[previous_valid_index, column] + df.loc[next_valid_index, column]) / 2
    
    df.set_index('datetime', inplace=True)
    return df
