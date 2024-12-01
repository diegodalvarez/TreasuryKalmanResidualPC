#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 02:31:24 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd
from   PCASignal import SignalBacktest

import matplotlib.pyplot as plt

class Backtest(SignalBacktest):
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.backtest_path = os.path.join(self.data_path, "backtest")
        if os.path.exists(self.backtest_path) == False: os.makedirs(self.backtest_path)
        
        self.window = 30
        
    def get_avg_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.backtest_path, "YieldPCA.parquet")
        try:
            
            if verbose == True: print("Trying to find Yield PCA data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
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
    
    def _get_rolling_sharpe(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
        df_out = (df.sort_values(
            "date").
            assign(
                rolling_mean = lambda x: x.signal_rtn.rolling(window = window).mean(),
                rolling_std  = lambda x: x.signal_rtn.rolling(window = window).std(),
                sharpe       = lambda x: x.rolling_mean / x.rolling_std,
                lag_sharpe   = lambda x: x.sharpe.shift()).
            drop(columns = ["rolling_mean", "rolling_std", "sharpe"]))
        
        return df_out
    
    def get_rolling_sharpe(self, window: int = 30) -> pd.DataFrame: 
        
        df = (self.get_zscore().drop(
            columns = ["PX_bps"]).
            assign(
                group_var = lambda x: 
                    x.security + " " + x.input_var + " " + 
                    x.variable + " " + x.window).
            groupby("group_var").
            apply(self._get_rolling_sharpe, window).
            reset_index(drop = True).
            dropna().
            drop(columns = ["group_var"]))
        
        return df
    
    def _get_max_sharpe(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.query(
            "lag_sharpe == lag_sharpe.max()").
            head(1))
        
        return df_out
    
    def get_max_sharpe(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.backtest_path, "Backtest.parquet")
        try:
            
            if verbose == True: print("Trying to find optimized data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data, collecting")
            df_out = (self.get_rolling_sharpe().groupby([
                "date", "security", "variable", "input_var"]).
                apply(self._get_max_sharpe).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _rolling_vol(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                roll_std = lambda x: x.signal_rtn.rolling(window = window).std(),
                lag_std  = lambda x: x.roll_std.shift()).
            drop(columns = ["roll_std"]).
            dropna())
        
        return df_out
    
    def _get_erc(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df[
            ["date", "lag_std"]].
            groupby("date").
            agg("sum").
            rename(columns = {"lag_std": "cum_std"}).
            merge(right = df, how = "inner", on = ["date"]).
            assign(weighted_rtn = lambda x: x.lag_std / x.cum_std * x.signal_rtn))
        
        return df_out
    
    def get_erc_portfolio(self, verbose: bool = False) -> pd.DataFrame:
        
        
        file_path = os.path.join(self.backtest_path, "ERCBacktest.parquet")
        try:
            
            if verbose == True: print("Trying to find optimized data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find ERC portoflios")
            
            df_roll_vol = (self.get_max_sharpe().assign(
                group_var = lambda x: x.security + " " + x.input_var + " " + x.variable).
                groupby("group_var").
                apply(self._rolling_vol, self.window).
                reset_index(drop = True).
                drop(columns = ["group_var"]))
            
            df_out = (df_roll_vol.assign(
                group_var = lambda x: x.input_var + " " + x.variable).
                groupby("group_var").
                apply(self._get_erc).
                reset_index(drop = True).
                rename(columns = {"group_var": "port"}))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

def main() -> None: 
    
    df = Backtest().get_erc_portfolio(verbose = True)
    df = Backtest().get_max_sharpe(verbose = True)
    df = Backtest().get_avg_rtn(verbose = True)

if __name__ == "__main__": main()             
    