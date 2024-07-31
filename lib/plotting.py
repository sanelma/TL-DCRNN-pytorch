
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates as mdates
import datetime
import sys



def plot_dcrnn_realtime(df_test, y_true, y_preds, steps_into_future, val_ratio, test_ratio, 
                        sensor_ids,
                        num_samples, seq_len, horizon, time_stamps = 288 *4,
                        y_baseline = None,
                        cluster = None,
                        save_plots = True, 
                        show_plots = True):
    
  """""
  Plots the true and predicted values for a specified numbers of hours from a specified starting time
     
  :param df_test: 
  - time_stamps: number of time stamps to plot
  - cluster: only used to display cluster ID in title
  - y_baseline: VAR baseline data to plot as a comparison 
  """
  pred_interval = 60/seq_len
  min_forecasted = int(pred_interval*(steps_into_future+1))

  df_test.index = pd.to_datetime(df_test.index)
  df_test = df_test.loc[:, sensor_ids]
  
  # y_true is of size (horizon, time_stamps, n_nodes)
  y_true = y_true[steps_into_future, :time_stamps, :]
  y_preds_hour = y_preds[11, :time_stamps, :]
  y_preds = y_preds[steps_into_future, :time_stamps, :]

  if y_baseline is not None:
     y_baseline = y_baseline[steps_into_future, :time_stamps, :]
   

  train_ratio = 1 - val_ratio - test_ratio

  #n = n_val/ val_ratio
  n_val = round(num_samples*val_ratio)
  n_train = round(num_samples*train_ratio) 
  n_test = round(num_samples * test_ratio) 


  # start_index in speed data. 
  # note: these are assuming you use test data 
  start_index = int(n_train) + int(n_val) + seq_len + steps_into_future
  end_index = int(n_train) + int(n_val) + seq_len + time_stamps + steps_into_future

  nrows = 4
  ncols = 1
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12), sharey=True)
  axes = axes.flatten()  # Flatten the array to make indexing easier
  warnings.filterwarnings('ignore')

  for i in range(4):
      
      j = i
  #    j = int(i*10) # plot sensors 0, 10, 20, 30
    
      
     # sensor = 'ig11FD104_D16'
      sensor = "K426.D26"
      j = df_test.columns.get_loc(sensor)
     #  set j based on value of i
        
      if i == 0:
         j = df_test.columns.get_loc("K83.D14")
     #   j = 20
    #     j = 126
      elif i == 1:
         j = df_test.columns.get_loc("K405.D28")
    #     j = 122
      elif i == 2:
          j = df_test.columns.get_loc("K83.D13")
    #     j = 98
      elif i == 3:
         j = 109
     
    
    

      # get true values from df_test and from outputs to ensure that they align
      
      y_true_current = df_test.iloc[start_index:end_index,j:j+1]

      y_true_output = y_true[:, j]

      # predictions
      y_preds_temp = y_preds[:, j]
      y_preds_hour_temp = y_preds_hour[:, j]
      if y_baseline is not None:
         y_baseline_temp = y_baseline[:, j]

      ##### check to make sure true values from model and from df test align
      # if they don't, there is an error with the indexing, the cluster ids, or something
      y_true_current_list = [round(float(num), 2) for num in y_true_current.iloc[:, 0].tolist()]
      y_true_output_list = [round(float(num), 2) for num in y_true_output]


         

      if  y_true_current_list != y_true_output_list:
          print('True values do not align')
          print(y_true_current)
          print(y_true_output)
    #      sys.exit("Error with indexing or cluster ids")  

      # add time series indexing to outputs
      y_preds_temp = pd.Series(y_preds_temp, index=y_true_current.index) # add time series indexing 
      y_preds_hour_temp = pd.Series(y_preds_hour_temp, index=y_true_current.index) # add time series indexing
      y_true_output = pd.Series(y_true_output, index=y_true_current.index)

      # if baseline was given, we plot it as well
      if y_baseline is not None:
         y_baseline_temp = pd.Series(y_baseline_temp, index=y_true_current.index) # add time series indexing
         axes[i].plot(y_baseline_temp, label='VAR predictions', marker='', color = 'green')




      # Plot true values and predictions
    

      axes[i].plot(y_true_output, label='True counts', marker='', color = 'red')
    #  axes[i].plot(y_preds_hour_temp, label= "hour", marker = '', color = "orange")
      axes[i].plot(y_preds_temp, label= "DCRNN Predictions", marker = '', color = "blue")

                  
      # Set titles, labels, formats
      axes[i].set_title(f'Sensor {df_test.columns[j]}')
      axes[i].set_ylabel('Count')
      labels = axes[i].get_xticklabels()
      axes[i].set_xticklabels(labels = labels, rotation=45)
      axes[i].tick_params(axis='y', which='both', labelleft=True)
      axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Formatter to display hours and minutes

  # Adjust layout to prevent overlap
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.xticks(rotation=45) 
  axes[0].legend()
  fig.suptitle(f'Prediction {min_forecasted} minutes ahead from cluster {cluster}', fontsize=16, fontweight='bold')

  if save_plots:
      current_time = datetime.datetime.now()
      formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
      plt.savefig(f'plots/plot_realtime{formatted_time}.png', format='png', dpi=300)  # Adjust the filename and format as needed
  if show_plots:
    plt.show()



def plot_hourly(df_test, y_true, y_preds,  val_ratio, test_ratio, 
                sensor_ids, num_samples, seq_len, horizon, time_stamps = 288*4,
                y_baseline = None, hours = 8, cluster = None, save_plots = True, show_plots = True):
        
  # next want to plot hour to hour
  # first index is which time stamp of 12. i want to always plot all of these




  df_test = df_test.loc[:, sensor_ids]

  train_ratio = 1 - val_ratio - test_ratio

  #n = n_val/ val_ratio
  n_val = round(num_samples*val_ratio)
  n_train = round(num_samples*train_ratio) 
  n_test = round(num_samples * test_ratio) 



  # start_index in speed data. 
  # note: these are assuming you use test data 
  start_index = int(n_train) + int(n_val) + seq_len
  end_index = int(n_train) + int(n_val) + seq_len + n_test

  nrows = 4
  ncols = 1
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12), sharey=True)
  axes = axes.flatten()  # Flatten the array to make indexing easier
  warnings.filterwarnings('ignore')

  y_preds = np.array(y_preds)
  y_true = np.array(y_true)

  for i in range(4):
      
      y_true_output = []
      y_preds_temp = []
      y_preds_baseline = []
      j = i
      if i == 0:
         j = df_test.columns.get_loc("K83.D14")
     #   j = 20
    #     j = 126
      elif i == 1:
         j = df_test.columns.get_loc("K405.D28")
    #     j = 122
      elif i == 2:
          j = df_test.columns.get_loc("K83.D13")
    #     j = 98
      elif i == 3:
         j = 109
     

      # here, can plot custom sensors
    #  j = int(i*10)
   #   if i == 0:
     #   j = 20
  #      j = 126
   #   elif i == 1:
   #      j = 122
   #   elif i == 2:
   #      j = 98
   #   elif i == 3:
   #      j = 109
   
      
    #  sensor = 'ig11FD104_D16'
      
#      j = df_test.columns.get_loc(sensor)

      
      for hour in range(hours):

          # predictions
          
          y_preds_hour = y_preds[:, hour*horizon:hour*horizon+1, j]
          y_true_hour = y_true[:, hour*horizon:hour*horizon+1, j]

          y_true_output.append(y_true_hour)
          y_preds_temp.append(y_preds_hour)

          if y_baseline is not None:
            y_baseline_hour = y_baseline[:, hour*horizon:hour*horizon+1, j]
            y_preds_baseline.append(y_baseline_hour)

      y_true_output = np.concatenate(y_true_output).squeeze()
      y_preds_temp = np.concatenate(y_preds_temp).squeeze()

      if y_baseline is not None:
         y_preds_baseline = np.concatenate(y_preds_baseline).squeeze()


      # get true values from df_test and from outputs to ensure that they align
      y_true_current = df_test.iloc[start_index:start_index+(hours)*horizon,j:j+1]


      ##### check to make sure true values from model and from df test align
      # if they don't, there is an error with the indexing, the cluster ids, or something
      y_true_current_list = [round(float(num), 2) for num in y_true_current.iloc[:, 0].tolist()]
      y_true_output_list = [round(float(num), 2) for num in y_true_output]

      if  y_true_current_list != y_true_output_list:
          print('True values do not align')
          print(y_true_current)
          print(y_true_output)
         # sys.exit("Error with indexing or cluster ids")  

      # add time series indexing to outputs
      y_preds_temp = pd.Series(y_preds_temp, index=y_true_current.index) # add time series indexing 
      y_true_output = pd.Series(y_true_output, index=y_true_current.index)


      # if baseline was given, we plot it as well
      if y_baseline is not None:
         y_preds_baseline = pd.Series(y_preds_baseline, index=y_true_current.index) # add time series indexing
         axes[i].plot(y_preds_baseline, label='VAR Predictions', marker='', color = 'green')




      # Plot true values and predictions
      axes[i].plot(y_true_output, label='True counts', marker='', color = 'red')
      axes[i].plot(y_preds_temp, label= "DCRNN Predictions", marker = '', color = "blue")

                      
      # Set titles, labels, formats
      axes[i].set_title(f'Sensor {df_test.columns[j]}')
      axes[i].set_ylabel('Count')
      labels = axes[i].get_xticklabels()
      axes[i].set_xticklabels(labels = labels, rotation=45)
      axes[i].tick_params(axis='y', which='both', labelleft=True)
      axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Formatter to display hours and minutes

  # Adjust layout to prevent overlap
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.xticks(rotation=45) 
  axes[0].legend()
  fig.suptitle(f'Hourly predictions from cluster {cluster}', fontsize=16, fontweight='bold')
  
  if save_plots:
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'plots/plot_hourly{formatted_time}.png', format='png', dpi=300)  # Adjust the filename and format as needed
  if show_plots:
    plt.show()


def generate_weekday_weekend_averages(speed_data):
    
    speed_data.index = pd.to_datetime(speed_data.index)

    speed_data['day'] = speed_data.index.dayofweek
    speed_data['time'] = speed_data.index.time

    # split weekdays from weekends
    speed_data_weekday = speed_data[speed_data['day'] < 5]
    speed_data_weekend = speed_data[speed_data['day'] >= 5]

    weekday_averages = get_averages_over_day(speed_data_weekday)
    weekend_averages = get_averages_over_day(speed_data_weekend)

    return weekday_averages, weekend_averages


def get_averages_over_day(speed_df):

    speed_df = speed_df.groupby('time').mean().mean(axis=1)
    # reformat as dataframe
    speed_df = pd.DataFrame(speed_df)
    speed_df.reset_index(inplace=True)
    speed_df.columns = ['time', 'avg']
    speed_df['time'] = pd.to_datetime(speed_df['time'], format='%H:%M:%S').dt.time

    return speed_df


def plot_averages_over_day(speed_data, target_speed_data, 
                           day = "weekends", 
                           measure = "Counts", 
                           source = "Zurich", 
                           target = "Lucerne", 
                           save_plot = False):
    
    """""
    Plots the average counts or speeds over the course of day for the source and target datasets
     
    :Inputs: 
    - speed_data: average counts or speeds by time period, output from generate_weekday_weekend_averages
    - target_speed_data: average counts or speeds by time period, output from generate_weekday_weekend_averages
    Remaining inputs are only for graph labeling:
    :param day: weekdays or weekends
    :param measure: Counts or Speeds
    :param source: Source city/area name
    :param target: Source city/area name  
    """

    assert day in ['weekends', 'weekdays', 'both']
    assert measure in ['Counts', 'Speeds']
    if measure == 'Counts':
        unit = 'Vehicles/hour'
    elif measure == 'Speeds':
        unit = 'Miles/hour'

    ax = speed_data.plot(x=str('time'), y='avg', figsize=(15, 6), color = "black")
    target_speed_data.plot(x=str('time'), y='avg', ax=ax, color = "blue")

    plt.suptitle(f'Average counts over the day, {day}', size = 20)
    plt.title("Computed over all sensors", size = 18)

    # set x axis labels
    plt.xticks(labels = [f"{x}:00" for x in range(24)], 
               ticks=[f"{x}:00" for x in range(24)],
               rotation=45, 
               size = 14)
    #plt.ylim(0, 275)
    plt.yticks(size = 14)

    plt.xlabel("Time", size = 18)
    plt.ylabel(f'{measure} ({unit})', size = 18)
    ax.legend(labels=[f'Source ({source})', f'Target ({target})'], fontsize=14)
    
    if save_plot:
        plt.savefig(f'plots/{day}_{measure}_{source}.png', format='png', dpi=300)

    plt.show()
    


