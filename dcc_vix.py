# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:39:09 2020

@author: t656703
"""

import mgarch
import pandas as pd
import pdb

def getdata():
    data=pd.read_excel(filename, header=0, sheet_name='SP 500', usecols=['VIX','SP500.Log.Returns'] )
    data['factor 2']=data['SP500.Log.Returns']
   
    return data.iloc[1:,:]

if __name__ == '__main__':
    
    filename=r"\\ubsprod.msad.ubs.net\userdata\t656703\home\Documents\R\GARCH\SPXVIX.xlsx"
    wb_path=r"\\ubsprod.msad.ubs.net\userdata\t656703\home\Documents\distributionComparison\comparison_values.xlsx"
    data=getdata()
    ret=data[['factor 2', 'SP500.Log.Returns']]
    c=data['VIX']
    g_model=mgarch.mgarch()
    res=g_model.fit(ret, c)