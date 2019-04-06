
Name="Ravikrishnan"
sd=pd.read_excel(r"C:\Users\Prem Prasad\Desktop\Mayank-Fitbit\merged_Whole Data\{}.xlsx".format(Name),
                 sheet_name="Heart",index=False)
sleep = sd.iloc[:,[1,2,12]]  # 1 for Date, 2 for time, 12 for Sleep
unique_dates = []
for d in range(0,len(sleep)):
    unique_dates.append(str.split(str(sleep.iloc[d][0]))[0])
unique_dates = list(set(unique_dates))
unique_dates.sort()
temp = unique_dates
cols = ['Date','Night Sleep','Night 1s','Night 2s','Night 3s', 'Day Sleep','Day 1s','Day 2s','Day 3s'] 
df_day_night = pd.DataFrame(columns = cols)
i = 0

for k in unique_dates:
    
    cur_date = k
    print(k)
    night_sleep = 0
    night_1s = 0
    night_2s = 0
    night_3s = 0
    
    day_sleep = 0
    day_1s = 0
    day_2s = 0
    day_3s = 0
    
    
    while(str.split(str(sleep.iloc[i][0]))[0] == cur_date):
        
        time_check = str(sleep.iloc[i][1])
        
        if time_check < '06:00:00' or time_check > '22:00:00':
            if sleep.loc[i,'Sleep']==1:
                night_1s = night_1s + 1
            
            if sleep.loc[i,'Sleep']==2:
                night_2s = night_2s + 1
            
            if sleep.loc[i,'Sleep']==3:
                night_3s = night_3s + 1
        
        else:
            if sleep.loc[i,'Sleep']==1:
                day_1s = day_1s + 1
            
            if sleep.loc[i,'Sleep']==2:
                day_2s = day_2s + 1
            
            if sleep.loc[i,'Sleep']==3:
                day_3s = day_3s + 1
        i = i + 1
    
    night_sleep = night_1s + night_2s + night_3s
    day_sleep = day_1s + day_2s + day_3s
    
    row = [cur_date, night_sleep, night_1s, night_2s, night_3s, day_sleep, day_1s, day_2s, day_3s ]
    
    df_day_night = df_day_night.append(pd.Series(row,index = cols),ignore_index = True)
writer = pd.ExcelWriter('{} Sleep_Pattern_Day_Night.xlsx'.format(Name))
df_day_night.to_excel(writer,'Sheet1')
#df2.to_excel(writer,'Sheet2') 
writer.save()
df_month = []

for date in df_day_night['Date']:
    df_month.append(date.split("-")[1]) 


Dec = []
Jan = []
Feb = []
for i in df_month:
    if i == '12':
        Dec.append(True)
        Jan.append(False)
        Feb.append(False)
    if i == '01':
        Dec.append(False)
        Jan.append(True)
        Feb.append(False)
    if i == '02':
        Dec.append(False)
        Jan.append(False)
        Feb.append(True)
df_Dec = df_day_night[Dec]
df_Jan = df_day_night[Jan]
df_Feb = df_day_night[Feb]
df_subplot_Dec = df_Dec.loc[:,['Date','Night Sleep','Day Sleep']]
df_subplot_Jan = df_Jan.loc[:,['Date','Night Sleep','Day Sleep']]
df_subplot_Feb = df_Feb.loc[:,['Date','Night Sleep','Day Sleep']]
df_subplot_Dec.set_index('Date',inplace=True)
dec_plot = df_subplot_Dec.plot.bar(legend=False,title = '{} Day-Night Sleep in Dec'.format(Name))
fig = dec_plot.get_figure()
fig.savefig('{} December Day-Night Sleep.png'.format(Name), dpi = 900)
writer = pd.ExcelWriter('{} December Day-Night Sleep.xlsx'.format(Name))
df_subplot_Dec.to_excel(writer,'Sheet1')
writer.save()

# January
df_subplot_Jan.set_index('Date',inplace=True)
jan_plot = df_subplot_Jan.plot.bar(legend=False,title = '{} Day-Night Sleep in Jan'.format(Name))
fig = jan_plot.get_figure()
fig.savefig('{} January Day-Night Sleep.png'.format(Name), dpi = 900)
writer = pd.ExcelWriter('{} January Day-Night.xlsx'.format(Name))
df_subplot_Jan.to_excel(writer,'Sheet1')
writer.save()  

# February
df_subplot_Feb.set_index('Date',inplace=True)
feb_plot = df_subplot_Feb.plot.bar(legend=False,title = '{} Day-Night Sleep in Feb'.format(Name))
fig = feb_plot.get_figure()
fig.savefig('{} February Day-Night Sleep.png'.format(Name), dpi = 900)
writer = pd.ExcelWriter('{} February Day-Night Sleep.xlsx'.format(Name))
df_subplot_Dec.to_excel(writer,'Sheet1')
writer.save()