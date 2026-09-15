# EXPERIMENT 1
# TIME SERIES DECOMPOSITION OF NETWORK TRAFFIC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

 
# STEP 1: Create dates
 
dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

 
# STEP 2: Create a TREND
 
trend = np.linspace(1000, 1800, 120)

 
# STEP 3: Create a SEASONAL pattern
 
seasonal = 200 * np.sin(2 * np.pi * np.arange(120) / 7)

 
# STEP 4: Create random NOISE
 
np.random.seed(10)
noise = np.random.normal(0, 50, 120)
 
# STEP 5: Create network traffic
 
traffic = trend + seasonal + noise

 
# STEP 6: Create a DataFrame
 
data = pd.DataFrame({
    "Date": dates,
    "Network_Traffic": traffic
})

 
# STEP 7: Make Date the index
 
data.set_index("Date", inplace=True)

 
# STEP 8: Display some data # ---------------------------------------------------
print("NETWORK TRAFFIC DATA")
print("--------------------")
print(data.head())
print("\nTotal number of records:", len(data))

 
# STEP 9: Plot original network traffic
 
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Network_Traffic"])
plt.title("Original Network Traffic")
plt.xlabel("Date")
plt.ylabel("Network Traffic")
plt.grid(True)
plt.show()

 
# STEP 10: Perform time series decomposition
 
result = seasonal_decompose(data["Network_Traffic"], model="additive", period=7)

 # STEP 11: Display decomposition
 
result.plot()
plt.suptitle("Time Series Decomposition of Network Traffic", fontsize=14)
plt.show()

 
# STEP 12: Display the components
 
print("\nDecomposition completed successfully!")
print("\nTrend:")
print(result.trend.head())
print("\nSeasonal:")
print(result.seasonal.head())
print("\nResidual:")
print(result.resid.head())
