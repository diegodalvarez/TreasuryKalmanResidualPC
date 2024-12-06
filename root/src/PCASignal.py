#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:19:08 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd

from pykalman import KalmanFilter
from DataCollect import DataCollect
from sklearn.decomposition import PCA

class SignalBacktest(DataCollect):
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.signal_path = os.path.join(self.data_path, "Signals")
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
        self.n_components = 3
        self.z_score_windows = [5, 10, 20]


    def process_yld(self) -> pd.DataFrame: 
        
        df_out = (self.get_tsy_rate()[
            ["date", "variable", "value"]].
            rename(columns = {"value": "yld"}).
            assign(log_yld = lambda x: np.log(x.yld)).
            rename(columns = {"variable": "tenor"}).
            melt(id_vars = ["date", "tenor"]).
            rename(columns = {"variable": "input_var"}))
        
        return df_out
    
    def _get_pca(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame: 
        
        df_wider = (df.drop(
            columns = ["input_var"]).
            pivot(index = "date", columns = "tenor", values = "value").
            dropna())
        
        pca_model = PCA(n_components = n_components)
        
        df_fitted = (pd.DataFrame(
            data    = pca_model.fit_transform(df_wider),
            columns = ["PC{}".format(i + 1) for i in range(n_components)],
            index   = df_wider.index).
            reset_index().
            melt(id_vars = ["date"]).
            rename(columns = {"value": "fitted_val"}))
        
        df_exp_var = (pd.DataFrame(
            data    = pca_model.fit(df_wider).explained_variance_ratio_,
            columns = ["exp_var_ratio"],
            index   = ["PC{}".format(i + 1) for i in range(n_components)]).
            reset_index().
            rename(columns = {"index": "variable"}))
        
        df_out = (df_fitted.merge(
            right = df_exp_var, how = "inner", on = ["variable"]))
        
        return df_out
        
    def get_yld_pca(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "YieldPCA.parquet")
        try:
            
            if verbose == True: print("Trying to find Yield PCA data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Couldn't find data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
        
            df_out = (self.process_yld().groupby(
                "input_var").
                apply(self._get_pca, self.n_components).
                reset_index().
                drop(columns = ["level_1"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_fut_pca(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.signal_path, "FuturesPCA.parquet")
        try:
            
            if verbose == True: print("Trying to find Treasury PCA data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Couldn't find data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it")
            
            df_tmp = self.get_tsy_fut()
            df_out = (df_tmp[
                ["date", "security", "PX_bps"]].
                pivot(index = "date", columns = "security", values = "PX_bps").
                cumsum().
                reset_index().
                melt(id_vars = "date").
                dropna().
                merge(
                    right = df_tmp[["date", "security", "PX_pct"]], 
                    how   = "inner", 
                    on    = ["date", "security"]).
                rename(columns = {"value": "PX_bps"}).
                assign(security = lambda x: x.security.str.split(" ").str[0]).
                melt(id_vars = ["date", "security"]).
                rename(columns = {
                    "variable": "input_var",
                    "security": "tenor"}).
                groupby("input_var").
                apply(self._get_pca, self.n_components).
                reset_index().
                drop(columns = ["level_1"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_kalman(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp        = df.sort_values("date").dropna()
        kalman_filter = KalmanFilter(
            transition_matrices      = [1],
            observation_matrices     = [1],
            initial_state_mean       = 0,
            initial_state_covariance = 1,
            observation_covariance   = 1,
            transition_covariance    = 0.01)
        
        state_means, state_covariances = kalman_filter.filter(df_tmp.value)
        df_out = (df_tmp.assign(
            smooth     = state_means,
            lag_smooth = lambda x: x.smooth.shift(),
            resid      = lambda x: x.lag_smooth - x.value,
            lag_resid  = lambda x: x.resid.shift()).
            drop(columns = ["value"]))
        
        return df_out
    
    def prep_pca(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "KalmanSignals.parquet")
        try:
            
            if verbose == True: print("Trying to find Kalman Filters")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it")
        
            df_out = (pd.concat([
                self.get_fut_pca(), self.get_yld_pca()]).
                drop(columns = ["exp_var_ratio"]).
                assign(group_var = lambda x: x.input_var + " " + x.variable).
                rename(columns = {"fitted_val": "value"}).
                groupby("group_var").
                apply(self._get_kalman).
                reset_index(drop = True))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                window       = lambda x: str(window),
                rolling_mean = lambda x: x.lag_resid.ewm(span = window, adjust = False).mean(),
                rolling_std  = lambda x: x.lag_resid.ewm(span = window, adjust = False).std(),
                z_score      = lambda x: (x.lag_resid - x.rolling_mean) / x.rolling_std,
                lag_zscore   = lambda x: x.z_score.shift()).
            drop(columns = ["rolling_mean", "rolling_std", "z_score"]).
            dropna())
        
        return df_out
    
    def _get_zscore(self, df: pd.DataFrame, windows: list) -> pd.DataFrame: 
        
        df_out = (pd.concat([
            self._zscore(df, window)
            for window in windows]))
        
        return df_out
    
    def get_zscore(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "KalmanResidualZScore.parquet")
        try:
            
            if verbose == True: print("Trying to find ZScores Kalman Residuals")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except:
            
            if verbose == True: print("Couldn't find data, collecting it")
            df_zscore = (self.prep_pca()[
                ["input_var", "date", "variable", "lag_resid"]].
                dropna().
                assign(group_var = lambda x: x.input_var + " " + x.variable).
                groupby("group_var").
                apply(self._get_zscore, self.z_score_windows).
                reset_index(drop = True).
                drop(columns = ["group_var", "lag_resid"]))
            
            df_tmp = (self.get_tsy_fut()[
                ["date", "security", "PX_bps"]].
                merge(right = df_zscore, how = "inner", on = ["date"]).
                assign(selector = lambda x: x.input_var.str.split("_").str[-1]))
            
            df_yld = (df_tmp.query(
                "selector == 'yld'").
                drop(columns = ["selector"]).
                assign(
                    signal     = lambda x: np.where(x.variable == "PC2", x.lag_zscore, -1 * x.lag_zscore),
                    signal_rtn = lambda x: np.sign(x.signal) * x.PX_bps,
                    security   = lambda x: x.security.str.split(" ").str[0]).
                drop(columns = ["lag_zscore"]))
            
            df_px = (df_tmp.query(
                "selector != 'yld'").
                drop(columns = ["selector"]).
                assign(
                    signal     = lambda x: np.where(x.variable == "PC1", -1 * x.lag_zscore, x.lag_zscore),
                    signal_rtn = lambda x: np.sign(x.signal) * x.PX_bps,
                    security   = lambda x: x.security.str.split(" ").str[0]).
                drop(columns = ["lag_zscore"]))
            
            df_out = (pd.concat([
                df_px, df_yld]).
                dropna())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

def main() -> None:
        
    df = SignalBacktest().get_zscore(verbose = True)
    df = SignalBacktest().prep_pca(verbose = True)
    df = SignalBacktest().get_yld_pca(verbose = True)
    df = SignalBacktest().get_fut_pca(verbose = True)
    
if __name__ == "__main__": main()