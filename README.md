# TreasuryKalmanResidualPC
Treasury Principal Component Analysis Kalman Residual Mean Reversion

# Approach
In this case apply PCA to the yield curve and the returns and then using a Kalman Filter using the residuals and then trade based on those residuals. 

# Inputted Values to PCA
1. Yields on Treasuries
2. Returns of the Futures Themselve

# Todo
1. Fix PCA for backtest on treasury futures
2. imply signal direction from ols
3. out-of-sample Kalman residual
