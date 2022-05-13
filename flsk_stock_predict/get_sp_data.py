import yfinance as yf
import requests
import pandas as pd

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# grab all tickers and put them into a list
basket = payload[0]['Symbol'].tolist()

def stock_data_pull(history):
  """
  Download Stock data
  """
  stock_final = pd.DataFrame()
  # iterate over each symbol
  for i in basket:  
      
      # print the symbol which is being downloaded
      print( str(basket.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)  
      
      try:
          # download the stock price 
          stock = []
          stock = yf.download(i,period=history,interval = "1d",progress=False)
          # append the individual stock prices 
          if len(stock) == 0:
              None
          else:
              stock['Name']=i
              stock_final = stock_final.append(stock,sort=False)
      except Exception:
          None
  return stock_final

def fill_blanks(df,srl_num,range,value_variable,date_variable):
  """
  fills missing observations
  some stocks may not have the same number of observations
  """
  stage_df = df.copy()
  for comb in stage_df[srl_num].unique(): 
    temp = stage_df[stage_df[srl_num] == comb].copy()
    stage_df = stage_df[stage_df[srl_num] != comb] # remove existing series detail
    temp2 = range.merge(temp,how='left',on=date_variable)
    # since we are filling missing data, other fields will be missing to
    # value info will be filled by zero, for other fields, we simply forward fill
    # Then back fill for full missing value coverage    
    #temp2[value_variable].fillna(0,inplace=True)
    temp2.fillna(method='ffill',inplace=True)
    temp2.fillna(method='bfill',inplace=True)
    # replace with new, full data subset
    stage_df = stage_df.append(temp2,ignore_index=True)
  return stage_df

def pre_proc_df(stock_total):
  stock_total.reset_index(inplace=True)
  stock_total = stock_total[['Date','Close','Name']]

  # homogenous the samples in each stock
  date_range = stock_total.Date.drop_duplicates()
  date_range = pd.DataFrame(date_range)

  stock_ready = fill_blanks(stock_total,'Name',date_range,'Close','Date')
  stock_ready.sort_values(by=['Name','Date'],inplace=True)
  return stock_ready

def create_input_df(df,window):
  stock_pred = pd.DataFrame()
  for i, tckr in enumerate(df.Name.unique()):
    temp = df[df.Name == tckr].copy()
    if i == 0:
      stock_pred = temp.iloc[-window:,:]
    else:
      stock_pred = pd.concat([stock_pred, temp.iloc[-window:,:] ])
  return stock_pred




