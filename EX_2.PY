import pandas as pd
import matplotlib.pyplot as plt 
# EXPERIMENT: DDoS Attack Prediction using ARIMA

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. Load Dataset

df = pd.read_csv("Midterm_53_group.csv")

print("Dataset Loaded Successfully")
print("Shape of Dataset:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Columns:")
print(df.columns)


 # 3. Data Preprocessing
 
# Convert Time column to numeric
df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

# Remove missing time values
df = df.dropna(subset=["Time"])

# Convert time into seconds
df["Second"] = df["Time"].astype(int)

# Count number of packets per second
traffic = df.groupby("Second").size().reset_index(name="Packets")

print("\nPackets Per Second:")
print(traffic.head())

print("\nTotal Time Intervals:", len(traffic))

# 4. Visualize Network Traffic

plt.figure(figsize=(14, 5))

plt.plot(
    traffic["Second"],
    traffic["Packets"]
)

plt.title("Network Traffic - Packets Per Second")
plt.xlabel("Time (Seconds)")
plt.ylabel("Number of Packets")

plt.grid(True)
plt.show()


# 5. Prepare Time Series Data

# Packet count is our time series
ts = traffic["Packets"].astype(float)

# Check for missing values
print("\nMissing Values:", ts.isnull().sum())

# Fill missing values if any
ts = ts.fillna(0)

# 6. Train-Test Split
train_size = int(len(ts) * 0.80)

train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

print("\nTraining Data:", len(train))
print("Testing Data:", len(test))

# 7. Build ARIMA Model

print("\nTraining ARIMA Model...")

model = ARIMA(
    train,
    order=(5, 1, 0)
)

model_fit = model.fit()

print("\nARIMA Model Summary:")
print(model_fit.summary())

# 8. Forecast Future Traffic


forecast = model_fit.forecast(
    steps=len(test)
)

# Convert forecast to Series
forecast = pd.Series(
    forecast,
    index=test.index
)

print("\nForecasted Traffic:")
print(forecast.head())

# 9. Compare Actual vs Forecast

plt.figure(figsize=(14, 5))

plt.plot(
    test.index,
    test.values,
    label="Actual Traffic"
)

plt.plot(
    forecast.index,
    forecast.values,
    label="Forecast Traffic"
)

plt.title("Actual vs Forecasted Network Traffic")
plt.xlabel("Time")
plt.ylabel("Packets")

plt.legend()
plt.grid(True)

plt.show()

# 10. Calculate MAE and RMSE

mae = mean_absolute_error(
    test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast
    )
)

print("\nModel Evaluation")
print("-------------------------")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))


# 11. DDoS Detection Threshold

mean_traffic = train.mean()
std_traffic = train.std()

threshold = mean_traffic + (3 * std_traffic)

print("\nDDoS Detection Threshold:")
print(round(threshold, 2))

# 12. Detect Potential DDoS Attacks

ddos_alerts = forecast[forecast > threshold]

print("\nPotential DDoS Alerts:")
print(ddos_alerts)

# 13. Display DDoS Result

if len(ddos_alerts) > 0:

    print("\n⚠ Potential DDoS Attack Detected!")

    print(
        "Number of Alert Intervals:",
        len(ddos_alerts)
    )

else:

    print("\n✓ No Potential DDoS Attack Detected")

# 14. Plot DDoS Threshold


plt.figure(figsize=(14, 5))

plt.plot(
    test.index,
    test.values,
    label="Actual Traffic"
)

plt.plot(
    forecast.index,
    forecast.values,
    label="Forecast Traffic"
)

plt.axhline(
    y=threshold,
    linestyle="--",
    label="DDoS Threshold"
)

plt.title("DDoS Attack Prediction using ARIMA")
plt.xlabel("Time")
plt.ylabel("Packets")

plt.legend()
plt.grid(True)

plt.show()



# 15. Final Result


print("           FINAL RESULT")


print("Algorithm       : ARIMA")
print("Model           : ARIMA(5,1,0)")
print("Training Data   :", len(train))
print("Testing Data    :", len(test))
print("MAE             :", round(mae, 2))
print("RMSE            :", round(rmse, 2))
print("DDoS Threshold  :", round(threshold, 2))
print("DDoS Alerts     :", len(ddos_alerts))

if len(ddos_alerts) > 0:
    print("Status          : Potential DDoS Detected")
else:
    print("Status          : No Potential DDoS Detected")
 