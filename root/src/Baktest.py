#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 02:31:24 2024

@author: diegoalvarez
"""
import os
import pandas as pd
from   PCASignal import SignalBacktest
import matplotlib.pyplot as plt

class Backtest(SignalBacktest):
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.backtest_path = os.path.join(self.data_path, "backtest")
        if os.path.exists(self.backtest_path) == False: os.makedirs(self.backtest_path)
        
    def get_avg_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "YieldPCA.parquet")
        try:
            
            if verbose == True: print("Trying to find Yield PCA data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Couldn't find data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
            df_out =  (self.get_zscore()[
                ["date", "input_var", "variable", "signal_rtn"]].
                groupby(["date", "input_var", "variable"]).
                agg("mean").
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
            
            
#df = Backtest().get_avg_rtn()