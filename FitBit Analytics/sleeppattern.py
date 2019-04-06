# -*- coding: utf-8 -*-
"""
Created on Sat May  5 13:09:00 2018

@author: Prem Prasad
"""

for x in unique_dates:
    #splitting year,month,day
    cur_date = x

#    cur_year = str.split(cur_date,"-")[0]
#    cur_month = str.split(cur_date,"-")[1]
#    cur_day = str.split(cur_date,"-")[2]
    
    count_0_left = 0
    count_0_right = 0
    count_1_left = 0
    count_1_right = 0
    count_2_left = 0
    count_2_right = 0
    count_3_left = 0
    count_3_right = 0
    
    sleep_start_time = 0
    sleep_stop_time = 0

#    all_zero_start = True
#    all_zero_stop = 0
#    zero_count = 0
    zero_start = False
    morning_flag = True
    zero_count = 0
    
    total_time =0
    total_zero_count = 0
    total_one_count = 0
    
    sleep_type = 'N/A'
    wrong_end_time = 0
#    zero_list = []
#    l = []
    
    #check if whole day is 00000000000:
    if sleep.loc[i,'Sleep']==0:
        
        c, t = count_consec_zeros(i,cur_date)
        
        if t == '23:59:00' and str(sleep.iloc[i][1]) in ['00:00:00', str.split(start_datetime[Name])[1]]:
            if str(sleep.iloc[i][1]) == '00:00:00':
                    sleep_stop_time = '00:00:00'
                    sleep_start_time = t
            else:
                sleep_stop_time = str.split(start_datetime[Name])[1]
                sleep_start_time = t
            
            duration_left = 0
            count_1_left = 0
            count_2_left = 0
            count_3_left = 0
            duration_right = 0
            count_1_right = 0
            count_2_right = 0
            count_3_right = 0
            total_zero_count = total_zero_count + c
            total_time = total_time + c
            i = i + c
        
        else:
            
            while(str.split(str(sleep.iloc[i][0]))[0] == cur_date):
                # 111111111111000000000
                if morning_flag == True:
                    #counting 0's 1's 2's 3's for left  
        #            if sleep.loc[i,'Sleep']==0:
        #                count_0_left = count_0_left + 1
        #                zero_count = 0
                    if sleep.loc[i,'Sleep']==1:
                        count_1_left = count_1_left + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==2:
                        count_2_left = count_2_left + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==3:
                        count_3_left = count_3_left + 1
                        zero_count = 0
                
                # *******00000111111111
                if morning_flag == False:
                    #counting 0's 1's 2's 3's for right
        #            if sleep.loc[i,'Sleep']==0:
        #                count_0_right = count_0_right + 1
        #                zero_count = 0
                    if sleep.loc[i,'Sleep']==1:
                        count_1_right = count_1_right + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==2:
                        count_2_right = count_2_right + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==3:
                        count_3_right = count_3_right + 1
                        zero_count = 0
                
                #for finding the sleep stop time at first half of day
                if sleep.loc[i,'Sleep']==0 and not zero_start:
                    zero_count, time_change = count_consec_zeros(i, cur_date)
        #            zero_list = zero_list.append(zero_count)
                    
                    #1111111111110000000000
                    if time_change == '23:59:00':
                        sleep_stop_time = str(sleep.iloc[i-1,1])
                        sleep_start_time = '23:59:00'
                        duration_left = count_1_left + count_2_left + count_3_left
                        count_1_left = count_1_left
                        count_2_left = count_2_left
                        count_3_left = count_3_left
                        duration_right = 0
                        count_1_right = 0
                        count_2_right = 0
                        count_3_right = 0
                        total_zero_count = total_zero_count + zero_count
                        total_time = total_time + zero_count
                        i = i + zero_count
                        break
                    
                    if zero_count > 30:
                        sleep_stop_time= str(sleep.iloc[i,1])
                        zero_start = True
                        morning_flag = False

# *****************  if time_change (check 26 Feb 2018 001111111111000)
                #11111111100000000000011111111   
                #for finding the sleep start time at second half of day
                if sleep.loc[i,'Sleep']!=0 and zero_start:
                    sleep_start_time= str(sleep.iloc[i,1])
                    zero_start = False
                
                total_time = total_time + 1
                i = i+1
#        if sleep.loc[i,'Sleep']==0:
#            sleep_stop_time = str(sleep.iloc[i-1,1])
#            k = i
#            while(sleep.loc[k,'Sleep']==0):
#                zero_count += 1
    else:
        while(str.split(str(sleep.iloc[i][0]))[0] == cur_date):
                # 111111111111000000000
                if morning_flag == True:
                    #counting 0's 1's 2's 3's for left  
        #            if sleep.loc[i,'Sleep']==0:
        #                count_0_left = count_0_left + 1
        #                zero_count = 0
                    if sleep.loc[i,'Sleep']==1:
                        count_1_left = count_1_left + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==2:
                        count_2_left = count_2_left + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==3:
                        count_3_left = count_3_left + 1
                        zero_count = 0
                
                # *******00000111111111
                if morning_flag == False:
                    #counting 0's 1's 2's 3's for right
        #            if sleep.loc[i,'Sleep']==0:
        #                count_0_right = count_0_right + 1
        #                zero_count = 0
                    if sleep.loc[i,'Sleep']==1:
                        count_1_right = count_1_right + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==2:
                        count_2_right = count_2_right + 1
                        zero_count = 0
                    
                    if sleep.loc[i,'Sleep']==3:
                        count_3_right = count_3_right + 1
                        zero_count = 0
                
                #for finding the sleep stop time at first half of day
                if sleep.loc[i,'Sleep']==0 and not zero_start:
                    zero_count , time_change = count_consec_zeros(i, cur_date)
        #            zero_list = zero_list.append(zero_count)
                    
                    #if last time is not 23:59:00 like date 29 Dec 2017
                    if str.split(str(sleep.iloc[i+zero_count][0]))[0] != cur_date and time_change not in ['00:00:00', '23:59:00']:
                        sleep_stop_time = str(sleep.iloc[i-1,1])
                        sleep_start_time = time_change
                        duration_left = count_1_left + count_2_left + count_3_left
                        count_1_left = count_1_left
                        count_2_left = count_2_left
                        count_3_left = count_3_left
                        duration_right = 0
                        count_1_right = 0
                        count_2_right = 0
                        count_3_right = 0
                        total_zero_count = total_zero_count + zero_count
                        total_time = total_time + zero_count
                        i = i + zero_count
                        break
                    
                    
                    #1111111111110000000000
                    if time_change == '23:59:00':
                        sleep_stop_time = str(sleep.iloc[i-1,1])
                        sleep_start_time = '23:59:00'
                        duration_left = count_1_left + count_2_left + count_3_left
                        count_1_left = count_1_left
                        count_2_left = count_2_left
                        count_3_left = count_3_left
                        duration_right = 0
                        count_1_right = 0
                        count_2_right = 0
                        count_3_right = 0
                        total_zero_count = total_zero_count + zero_count
                        total_time = total_time + zero_count
                        i = i + zero_count
                        break
                    
                    if zero_count > 30:
                        sleep_stop_time= str(sleep.iloc[i,1])
                        zero_start = True
                        morning_flag = False
                
                #11111111100000000000011111111   
                #for finding the sleep start time at second half of day
                if sleep.loc[i,'Sleep']!=0 and zero_start:
                    sleep_start_time= str(sleep.iloc[i,1])
                    zero_start = False
                
                total_time = total_time + 1
                i = i+1
    
    duration_left = count_1_left + count_2_left + count_3_left
    duration_right = count_1_right + count_2_right + count_3_right   
    
    total_zero_count = total_time - duration_left - duration_right
    
    # 11111111111000000000  (doesn't resume sleep for second half (night))
    if sleep_start_time == 0:
        sleep_start_time = '00:00:00'
    
    if sleep_stop_time == 0:
        sleep_stop_time = '00:00:00'
    
    if str(sleep.iloc[i-1][1])!= '23:59:00':
        wrong_end_time = 1
#    #sleeping full day:
#    if sleep_stop_time == '23:59:00' and sleep_start_time == '00:00:00':
#            sleep_stop_time = '23:59:00'
#    if sleep_stop_time == '00:00:00' and sleep_start_time == '23:59:00':
    if total_zero_count == total_time:
        sleep_type = '00000000000000'
    
    if sleep_stop_time == '00:00:00' and sleep_start_time!= '23:59:00':
        sleep_type = '00000001111111'
    
    if sleep_stop_time not in ['00:00:00',str.split(start_datetime[Name])[1]] and sleep_start_time == '23:59:00':
        sleep_type = '11111110000000'
    
    if sleep_stop_time != '00:00:00' and wrong_end_time == 1 and sleep.loc[i,'Sleep'] == 0:
        sleep_type = '11111110000000'
    
    if sleep_stop_time != '00:00:00' and sleep_start_time != '23:59:00' and sleep_stop_time < sleep_start_time:
        sleep_type = '11111000011111'
    
    if sleep_stop_time not in ['00:00:00',str.split(start_datetime[Name])[1]] and sleep_start_time != '23:59:00' and wrong_end_time == 1:
        sleep_type = '11111110000000'
    
    l = [cur_date, sleep_type, sleep_stop_time, sleep_start_time, wrong_end_time ,duration_left, count_1_left, count_2_left, count_3_left,
             total_zero_count, duration_right, count_1_right, count_2_right, count_3_right]
    
    df_sleep = df_sleep.append(pd.Series(l,index = col),ignore_index = True)
df_sleep