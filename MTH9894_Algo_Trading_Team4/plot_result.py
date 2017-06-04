
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import *
from datetime import datetime
from datetime import timedelta
import statsmodels.tsa.stattools as stat
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[14]:

def plot_result(result ):
    '''
    Plot result from backtesting: sp500 accumulative return and pair trading return, sharp ratio, maximum drawdown
    '''
    cash_value_list = np.array(result[0])
    sp500_index = np.array(result[1])
    sp500_ret = sp500_index / sp500_index[0] - 1.
    rf_rate = result[2]
    ret = cash_value_list / cash_value_list[0] - 1.
    
    #date_list = date_in_data[3000:4100]

   

    sharp_ratio = (np.mean(ret) - rf_rate) / sqrt(np.var(ret))

    max_here = pd.expanding_max(ret)
    dd2here = ret - max_here
    max_drawdown = dd2here.min()


    plt.figure(figsize = (16,5), dpi = 120)
    plt.plot(sp500_ret, color = "blue", label = "sp500")
    plt.plot(ret, color = "red", label = "pair trading")
    plt.annotate('Sharp Ratio ' + str(round(sharp_ratio,2)) + ', Max Drawdown '+ str(round(max_drawdown,2)), xy = (0.01, 0.8),xycoords = 'axes fraction')
    plt.legend()


# In[ ]:



