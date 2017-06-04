
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import *
from datetime import datetime
from datetime import timedelta
import statsmodels.tsa.stattools as stat


# In[10]:

def generate_signal(priceA, priceB, sa, sb, sc, sd):
    '''Find the trading signal in one window
    
    Parameters
    -----------
    priceA: numpy ndarray, price of stockA
    priceB: numpy ndarray, price of stockB
    sa    : float, the bound for controlling open long
    sb    : float, the bound for controlling close short and close position
    sc    : float, the bound for controlling close long and close position
    sd    : float, the bound for controlling open short
    
    Return
    -----------
    beta  : float, the proportion of stock A with respect to stock B
    signal: string, the direction should be taken for executing the order
    s_score: float'''
    
    assert len(priceA)==len(priceB),"Wrong data extraction!"
    signals = ['OL','OS','C','CS','CL','DONT']
    
    # calculate the stock return
    Areturn = np.diff(priceA)/priceA[:(len(priceA)-1)]
    Breturn = np.diff(priceB)/priceB[:(len(priceB)-1)]
    
    # regression on return
    beta, beta0, r_value, p_value, std_err = stats.linregress(Areturn, Breturn)
    
    assert beta!=0,'beta is zero!'
    assert isnan(beta)!=True,'beta is nan!'

    # get the residual epsilon_t
    e_t = np.array(Breturn - beta0 - beta*Areturn)

    Xt=[np.sum(e_t[:i+1]) for i in range(len(Areturn))]

    # regression on X_t
    length = len(Xt)
    Xt_vec = np.array(Xt)
    b, a, r_value_x, p_value_x, std_err_x = stats.linregress(Xt_vec[:length-1],Xt_vec[1:])

    # get the residual zeta_t
    z_t = Xt_vec[1:] - a - b*Xt_vec[:length-1]
    var_z = np.var(z_t)
    
    assert var_z>0,'divide by 0!'
    
    # calculate s_score
    if abs(b) >= 1:
        s_score = 100000
    else:
        s_score = -a*sqrt(1-b**2)/((1-b)*sqrt(var_z)) + int(a/(1-b))*sqrt((1-b**2)/var_z)
    
    # trading signal
    if s_score   < sa:                         # open long
        signal = signals[0]
    elif (s_score > sd) & (s_score < 100000):  # open short
        signal = signals[1]
    elif (s_score > sb) & (s_score <sc):       # close
        signal = signals[2]    
    elif (s_score > sa) & (s_score < sb):      # close short
        signal = signals[3]    
    elif (s_score > sc) & (s_score < sd):      # close long
        signal = signals[4]
    else:                                      # DONT
        signal = signals[5]
        
    return beta,signal,s_score


# In[11]:

position_long = {}  # static variable: long portfolio dict
position_short = {} # static variable: short portfolio dict
def build_position(cash,pairs, data, window, sa,sb,sc,sd):
    '''Build the position for updating position of pairs and portfolio value
    
    Parameters
    ----------
    cash  : float, static variable, update the cash amount we hold
    pairs : dict, store the pair being found in each window
    data  : dict, store the stock information for each pair stock
    window: int, the rolling length
    sa    : float, the bound for controlling open long
    sb    : float, the bound for controlling close short and close position
    sc    : float, the bound for controlling close long and close position
    sd    : float, the bound for controlling open short
    
    Return
    ----------
    cash  : float, static variable, update the cash amount
    cash_list: list, static variable, record the cash each day as a sum of all pairs
    '''
    
    # initialize parameters
    money=5000
    borrow_cost = 0.05
    transaction_cost = 0.003
    PNL = []
    portfolio_value = []
    cash_list = []
    
    stocks_in_pair=[]
    for pair in pairs:
        stocks_in_pair+=[pair[0]]
        stocks_in_pair+=[pair[1]]
    
    for ticker in position_long.keys():
        if ticker not in stocks_in_pair:
            if position_long[ticker] >0:
                cash += (data[ticker]['BID'].values[window] - transaction_cost)*position_long[ticker]
            else:
                cash += (data[ticker]['ASK'].values[window] + transaction_cost)*position_long[ticker]
    for ticker in position_short.keys():
        if ticker not in stocks_in_pair:
            if position_short[ticker] <0:
                cash += (data[ticker]['ASK'].values[window] + transaction_cost)*position_short[ticker]
            else:
                cash += (data[ticker]['BID'].values[window] - transaction_cost)*position_short[ticker]

    # for each day, calculate all pairs value change
    for i in range(window):
        stock_value = 0
        # for each pair
        for pair in pairs:
            # get data
            priceA        = data[pair[0]]['PRC'].values[i:i+window]
            priceB        = data[pair[1]]['PRC'].values[i:i+window]
            tickerA        = pair[0]
            tickerB        = pair[1]
            BidA           = data[pair[0]]['BID'].values
            AskA           = data[pair[0]]['ASK'].values
            BidB           = data[pair[1]]['BID'].values
            AskB           = data[pair[1]]['ASK'].values
            result         = generate_signal(priceA, priceB,sa,sb,sc,sd)
            beta           = result[0]
            signal         = result[1]
            s_score        = result[2]

            assert (beta!=0 and AskB[i+window]!=0 and BidB[i+window]!=0),'divide by 0!'
            if (isnan(AskB[i+window])==True or isnan(BidB[i+window])==True):
                print(tickerB)
                print(i+window)
                print(AskB,BidB)
            assert (isnan(AskB[i+window])==False and isnan(BidB[i+window])==False),'nan data!'
            
            # open long
            if signal=='OL':  # long B, short A
                position_long[tickerA]=position_long.get(tickerA,0)-int((money/len(pairs))/AskB[i+window]*beta)
                position_long[tickerB]=position_long.get(tickerB,0)+int((money/len(pairs))/AskB[i+window])
                cash += (BidA[i+window]-transaction_cost)*int((money/len(pairs))/AskB[i+window]*beta)
                cash-=(AskB[i+window]+transaction_cost)*int((money/len(pairs))/AskB[i+window])
            
            # open short
            elif signal=='OS':  # short B, long A
                position_short[tickerA]=position_short.get(tickerA,0)+int((money/len(pairs))/AskB[i+window]*beta)
                position_short[tickerB]=position_short.get(tickerB,0)-int((money/len(pairs))/AskB[i+window])
                cash -= (AskA[i+window]+transaction_cost)*int((money/len(pairs))/BidB[i+window]*beta)
                cash +=(BidB[i+window]-transaction_cost)*int((money/len(pairs))/BidB[i+window])
            
            # close position
            elif signal=='C': # set position to 0
                if tickerA in position_long.keys():
                    cash+=(BidA[i+window]-transaction_cost)*position_long[tickerA]
                    position_long[tickerA]=0
                if tickerA in position_short.keys():
                    cash+=(AskA[i+window]+transaction_cost)*position_short[tickerA]
                    position_short[tickerA]=0
                if tickerB in position_long.keys():
                    cash+=(BidB[i+window]-transaction_cost)*position_long[tickerB]
                    position_long[tickerB]=0
                if tickerB in position_short.keys():
                    cash+=(AskB[i+window]+transaction_cost)*position_short[tickerB]
                    position_short[tickerB]=0
            
            # close long       
            elif signal=='CL':
                if tickerA in position_long.keys():
                    cash+=(BidA[i+window]-transaction_cost)*position_long[tickerA]
                    position_long[tickerA]=0
                if tickerB in position_long.keys():
                    cash+=(BidB[i+window]-transaction_cost)*position_long[tickerB]
                    position_long[tickerB]=0
            
            # close short
            elif signal=='CS':
                if tickerA in position_short.keys():
                    cash+=(AskA[i+window]+transaction_cost)*position_short[tickerA]
                    position_short[tickerA]=0
                if tickerB in position_short.keys():
                    cash+=(AskB[i+window]+transaction_cost)*position_short[tickerB]
                    position_short[tickerB]=0 
        
        cash_list+=[cash]
          
    return cash,cash_list


# In[ ]:



