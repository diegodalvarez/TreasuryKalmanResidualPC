#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:07:21 2024

@author: diegoalvarez
"""
import os
import pandas as pd
import datetime as dt
import pandas_datareader as web

class DataCollect: 
    
    def __init__(self) -> None: 
        
        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path = os.path.join(self.repo_path, "data")
        self.raw_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)  == False: os.makedirs(self.raw_path)
        
        self.bbg_fut = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager"
        if os.path.exists(self.bbg_fut) == False: self.bbg_fut = r"/Users/diegoalvarez/Desktop/BBGFuturesManager"
        
        self.fut_tickers =  ["TU", "TY", "US", "FV", "UXY", "WN"]
        self.yld_tickers  = ["DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
        
        self.start_date = dt.date(year = 2000, month = 1, day = 1)
        self.end_date   = dt.date.today()
        
        
    def _get_yld_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        return(df.sort_values(
            "date").
            assign(val_diff = lambda x: x.value.diff()).
            dropna())
        
    def get_tsy_rate(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "FredTreasuryYields.parquet")

        try:
            
            if verbose == True: print("Trying to find FRED Treasury Data")
            df_tsy = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
            
            tickers = ["DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
            df_tsy = (web.DataReader(
                name        = tickers,
                data_source = "fred",
                start       = self.start_date,
                end         = self.end_date).
                reset_index().
                melt(id_vars = "DATE").
                dropna().
                rename(columns = {"DATE": "date"}).
                groupby("variable").
                apply(self._get_yld_diff).
                reset_index(drop = True).
                dropna())
            
            if verbose == True: print("Saving data\n")
            df_tsy.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_tsy
    
    def get_tsy_rate(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "FredTreasuryYields.parquet")

        try:
            
            if verbose == True: print("Trying to find FRED Treasury Data")
            df_tsy = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
            
            tickers = ["DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
            df_tsy = (web.DataReader(
                name        = tickers,
                data_source = "fred",
                start       = self.start_date,
                end         = self.end_date).
                reset_index().
                melt(id_vars = "DATE").
                dropna().
                rename(columns = {"DATE": "date"}).
                groupby("variable").
                apply(self._get_yld_diff).
                reset_index(drop = True).
                dropna())
            
            if verbose == True: print("Saving data\n")
            df_tsy.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_tsy
    
    def _get_fut_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        return(df.sort_values(
            "date").
            assign(
                PX_diff = lambda x: x.PX_LAST.diff(),
                PX_pct  = lambda x: x.PX_LAST.pct_change()).
            dropna())
    
    def get_tsy_fut(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "TreasuryFutures.parquet")
        try:
            
            if verbose == True: print("Trying to find Treasury Futures Data")
            df_fut = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found\n")
            
        except:
            
            if verbose == True: print("Couldn't find data now collecting it")
            
            px_paths = [
                os.path.join(self.bbg_fut, "data", "PXFront", file + ".parquet")
                for file in self.fut_tickers]
            
            deliv_paths = [
                os.path.join(self.bbg_fut, "data", "BondDeliverableRisk", file + ".parquet")
                for file in self.fut_tickers]
            
            df_px = (pd.read_parquet(
                path = px_paths, engine = "pyarrow").
                groupby("security").
                apply(self._get_fut_rtn).
                reset_index(drop = True))
            
            df_deliv = (pd.read_parquet(
                path = deliv_paths, engine = "pyarrow").
                pivot(index = ["date", "security"], columns = "variable", values = "value").
                reset_index().
                rename(columns = {
                    "FUT_EQV_CNVX_NOTL"            : "FUT_CNVX",
                    "CONVENTIONAL_CTD_FORWARD_FRSK": "CTD_DUR"}).
                dropna())
            
            df_fut = (df_px.merge(
                right = df_deliv, how = "inner", on = ["date", "security"]).
                assign(PX_bps = lambda x: x.PX_diff / x.CTD_DUR))
            
            df_fut.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_fut
    
def main() -> pd.DataFrame: 
        
    DataCollect().get_tsy_rate(verbose = True)
    DataCollect().get_tsy_fut(verbose = True)
    
if __name__ == "__main__": main()