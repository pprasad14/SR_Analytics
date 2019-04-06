# -*- coding: utf-8 -*-
"""
Created on Sat May  5 12:55:48 2018

@author: Prem Prasad
"""

Name = 'Tarun'
sleep_brief_df = pd.read_csv(r"C:\Users\Prem Prasad\Desktop\Mayank-Fitbit\temp\{} SleepBrief Collection.csv".format(Name))

sleep_time = sleep_brief_df['startTime']
sleep_count = {}

sleep_time_dec = []
sleep_time_jan = []
sleep_time_feb = []
sleep_time_mar = []
sleep_time_apr = []
sleep_time
for sleep in sleep_time:
    if type(sleep) == str:            
        if str.split(sleep.split("T")[0],"-")[1] == '12':
            sleep_time_dec.append(sleep)
        
        if str.split(sleep.split("T")[0],"-")[1] == '01':
            sleep_time_jan.append(sleep)
        
        if str.split(sleep.split("T")[0],"-")[1] == '02':
            sleep_time_feb.append(sleep)
        
        if str.split(sleep.split("T")[0],"-")[1] == '03':
            sleep_time_mar.append(sleep)
        
        if str.split(sleep.split("T")[0],"-")[1] == '04':
            sleep_time_apr.append(sleep)
    
    else:
        continue
dec_count = sleep_start_count(sleep_time_dec)
jan_count = sleep_start_count(sleep_time_jan)
feb_count = sleep_start_count(sleep_time_feb)
mar_count = sleep_start_count(sleep_time_mar)
apr_count = sleep_start_count(sleep_time_apr)
dec_count
dec_Key = [key for key, value in dec_count.items() if value == max(dec_count.values())][0]
dec_key_value = [dec_Key,max(dec_count.values())]

jan_Key = [key for key, value in jan_count.items() if value == max(jan_count.values())][0]
jan_key_value = [jan_Key,max(jan_count.values())]

feb_Key = [key for key, value in feb_count.items() if value == max(feb_count.values())][0]
feb_key_value = [feb_Key,max(feb_count.values())]

mar_Key = [key for key, value in mar_count.items() if value == max(mar_count.values())][0]
mar_key_value = [mar_Key,max(mar_count.values())]
df = pd.DataFrame(columns = ['Time', 'Frequency'])

df['Time'] = dec_count.keys()
df['Frequency'] = dec_count.values()
df = df.sort_values(by = ['Time'])
df = df.set_index("Time")

dec_plot = df.plot.bar(legend=False,fontsize = 5, title = "{} December Sleep Time Frequency".format(Name))

fig = dec_plot.get_figure()
fig.savefig('{} December Sleep Time Frequency.png'.format(Name), dpi = 900)
df = pd.DataFrame(columns = ['Time', 'Frequency'])

df['Time'] = jan_count.keys()
df['Frequency'] = jan_count.values()
df = df.sort_values(by = ['Time'])
df = df.set_index("Time")

jan_plot = df.plot.bar(legend=False,fontsize = 5, title = "{} January Sleep Time Frequency".format(Name))

fig = jan_plot.get_figure()
fig.savefig('{} January Sleep Time Frequency.png'.format(Name), dpi = 900)
df = pd.DataFrame(columns = ['Time', 'Frequency'])

df['Time'] = feb_count.keys()
df['Frequency'] = feb_count.values()
df = df.sort_values(by = ['Time'])
df = df.set_index("Time")

feb_plot = df.plot.bar(legend=False,fontsize = 5, title = "{} February Sleep Time Frequency".format(Name))

fig = feb_plot.get_figure()
fig.savefig('{} February Sleep Time Frequency.png'.format(Name), dpi = 900)
df = pd.DataFrame(columns = ['Time', 'Frequency'])

df['Time'] = mar_count.keys()
df['Frequency'] = mar_count.values()
df = df.sort_values(by = ['Time'])
df = df.set_index("Time")

mar_plot = df.plot.bar(legend=False,fontsize = 5, title = "{} March Sleep Time Frequency".format(Name))

fig = mar_plot.get_figure()
fig.savefig('{} March Sleep Time Frequency.png'.format(Name), dpi = 900)
