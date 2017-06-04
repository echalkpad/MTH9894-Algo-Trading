
# coding: utf-8

# In[10]:

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import *
from datetime import datetime
from datetime import timedelta
import statsmodels.tsa.stattools as stat
import pandas as pd


# In[11]:

def date_int_to_string(date_int):
    year = int(date_int / 10000)
    month = int((date_int - 10000 * year)/100)
    day = date_int - 10000*year - 100*month
    
    year_str = str(year)
    
    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)
    
    if day < 10:
        day_str = '0' + str(day)
    else:
        day_str = str(day)
    
    return year_str + '-' + month_str + '-' + day_str


# In[12]:

def data_process(start,period):
    with open ('sp_500_daily_prices.csv') as data:
        sp500=pd.read_csv(data,usecols=['date','COMNAM','PRC','RET','ASK','BID'],engine='c')
    date=sorted(list(set(sp500['date'])))
    try:
        date.index(start)
    except:
        raise (("Oops! The start date is not contained in our datesets.  Try again..."))
    try:
        train_start=date[-9*period+date.index(start)]
        date[date.index(start)+period]
    except:
        raise ("Oops! The period is too long.  Try again...")
    sp_recent=sp500[sp500['date']>=train_start]
    sp_recent=sp_recent[sp_recent['date']<=date[date.index(start)+period]]
    
    with open ('sp_500_daily_return.csv') as return_data:
        sp_ret=pd.read_csv(return_data,index_col=0,engine='c')
    sp_ret=sp_ret.loc[date[-9*period+date.index(start)]:date[period+date.index(start)]]
    return sp_recent,sp_ret,date


# In[13]:

def find_pair(start,period,num):
    sp_recent,sp_ret,date=data_process(start,period)
    error_return=['E','D','C','B','A','']
    sp_ret=sp_ret.applymap(lambda x:np.float64(0.0) if x in error_return else np.float64(x))
    sp_pca=np.array(sp_ret)
    #print(sp_pca)
    cov_company=np.cov(sp_pca)
    cov_company[np.isnan(cov_company)] = 0.0
    eigen_val,eigen_vector=np.linalg.eig(cov_company)
    eigen_vector_max=list(zip(eigen_vector[np.argmax(eigen_val)],sp_ret.columns))
    eigen_vector_max.sort(key=lambda x:x[0],reverse=True)
    eigen_vector_max=list(zip(*eigen_vector_max))
    data=sp_ret[list(eigen_vector_max[1][0:20])]
    pvalue_key=cointegration_test(data)
    selected_pair=filter_pair(pvalue_key,num)
    
    data_inuse=sp_recent[sp_recent['date']>=date[-period+date.index(start)]]
    data_inuse=data_inuse[data_inuse['date']<date[date.index(start)+period]]
    data_inuse.fillna(method='ffill',inplace=True)
    
    companies=set(data_inuse['COMNAM'])
    company_data={}
    for company in companies:
        company_data[company]=data_inuse[data_inuse['COMNAM']==company]
    #print('find_pair_done')
    return company_data,selected_pair


# In[14]:

def filter_pair(pvalue_key,num):
    try:
        pvalue_key[num-1]
    except:
        raise("Oops! Too large num")
    selected_pair=[]
    stock_list=[]
    i=0
    while i<num:
        for pair in pvalue_key:
            if pair[1][0] not in stock_list and pair[1][1] not in stock_list:
                selected_pair+=[pair[1]]
                stock_list+=[pair[1][0]]
                stock_list+=[pair[1][1]]
                i+=1
    
    return selected_pair


# In[15]:

def cointegration_test(data):
    n = len(data.columns)
    #score_matrix = np.zeros((n, n))
    pvalue_key=[]
    keys = list(data.columns)
    #pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = np.array(data[keys[i]])
            S2 = np.array(data[keys[j]])
            result = stat.coint(S1, S2)
            #score = result[0]
            pvalue = result[1]
            pvalue_key+=[[pvalue,(keys[i],keys[j])]]
            #pairs.append((keys[i], keys[j]))
    pvalue_key=sorted(pvalue_key,key=lambda x:x[0])
    return pvalue_key


# In[ ]:



