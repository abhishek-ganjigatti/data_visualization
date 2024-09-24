pip install ruptures
**#Maneuver Detection Tools
**
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('SMA_data.csv')
df.info()

df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Rolling Statistics (Window-Based Features)
df['rolling_mean'] = df['SMA'].rolling(window=3).mean()
df['rolling_std'] = df['SMA'].rolling(window=3).std()
df['rolling_min'] = df['SMA'].rolling(window=3).min()
df['rolling_max'] = df['SMA'].rolling(window=3).max()

# Linear Trend Feature
X = np.arange(len(df)).reshape(-1, 1)
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)
df['linear_trend'] = X_poly[:, 1]

# Exponential Moving Average
df['ema'] = df['SMA'].ewm(span=5, adjust=False).mean()

# Frequency Domain Features (Fourier Transform)
fft_values = fft(df['SMA'].values)
df['fft_real'] = np.real(fft_values)
df['fft_imag'] = np.imag(fft_values)

# Lag-Based Features
df['lag_1'] = df['SMA'].shift(1)
df['lag_2'] = df['SMA'].shift(2)

# Autoregressive Features
model = AutoReg(df['SMA'].dropna(), lags=2)
ar_model = model.fit()
df['ar_coef_1'] = ar_model.params[1]  # First AR coefficient
df['ar_coef_2'] = ar_model.params[2]  # Second AR coefficient

# Cumulative Features
df['cumulative_sum'] = df['SMA'].cumsum()
df['cumulative_max'] = df['SMA'].cummax()
df['cumulative_min'] = df['SMA'].cummin()

# Differencing and Change Features
df['diff_1'] = df['SMA'].diff(1)
df['diff_2'] = df['SMA'].diff(2)
df['pct_change'] = df['SMA'].pct_change()
df['direction'] = np.sign(df['SMA'].diff())

# Anomaly/Change Detection Features
df['z_score'] = (df['SMA'] - df['SMA'].mean()) / df['SMA'].std()
df['outlier'] = np.abs(df['z_score']) > 2  # Mark points as outliers if Z-score > 2

# Time-Based Features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Segment-Level Features (Segment by day)
df['date'] = df.index.date
daily_groups = df.groupby('date')['SMA']
df['daily_mean'] = daily_groups.transform('mean')
df['daily_std'] = daily_groups.transform('std')

# Drop temporary columns used for segmentation
df.drop(columns=['date'], inplace=True)

# ---- New Features ----

# SMA Velocity (First Derivative)
df['sma_velocity'] = df['SMA'].diff(1)  # First derivative of SMA

# Rolling Window SMA Differences (Difference between SMA and rolling mean)
df['rolling_diff'] = df['SMA'] - df['rolling_mean']



# Acceleration Features (Second Derivative)

df['sma_acceleration'] = df['sma_velocity'].diff(1)  # Second derivative of SMA

# Display the resulting DataFrame with extracted features
print(df.head())

**
#Visualizations
**

import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
sns.set(style="whitegrid")

# Plot SMA and other features
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
fig.suptitle('Maneuver Detection Visualizations', fontsize=16)

# Plot the SMA values over time
df['SMA'].plot(ax=axes[0, 0], title='SMA Over Time', color='blue')
axes[0, 0].set_ylabel('SMA')

# Plot the SMA velocity (first derivative)
df['sma_velocity'].plot(ax=axes[0, 1], title='SMA Velocity (First Derivative)', color='orange')
axes[0, 1].set_ylabel('Velocity')

# Plot the SMA acceleration (second derivative)
df['sma_acceleration'].plot(ax=axes[1, 0], title='SMA Acceleration (Second Derivative)', color='green')
axes[1, 0].set_ylabel('Acceleration')

# Plot the rolling mean of SMA
df['rolling_mean'].plot(ax=axes[1, 1], title='Rolling Mean of SMA', color='purple')
axes[1, 1].set_ylabel('Rolling Mean')

# Plot the rolling standard deviation of SMA
df['rolling_std'].plot(ax=axes[2, 0], title='Rolling Std Dev of SMA', color='red')
axes[2, 0].set_ylabel('Rolling Std Dev')

# Plot the rolling window SMA differences
df['rolling_diff'].plot(ax=axes[2, 1], title='Rolling Window SMA Differences', color='brown')
axes[2, 1].set_ylabel('Rolling Difference')

# Plot the exponential moving average (EMA) of SMA
df['ema'].plot(ax=axes[3, 0], title='Exponential Moving Average of SMA', color='cyan')
axes[3, 0].set_ylabel('EMA')

# Plot the cumulative sum of SMA
df['cumulative_sum'].plot(ax=axes[3, 1], title='Cumulative Sum of SMA', color='magenta')
axes[3, 1].set_ylabel('Cumulative Sum')

# Plot the z-score (anomaly detection)
df['z_score'].plot(ax=axes[4, 0], title='Z-Score for Anomaly Detection', color='grey')
axes[4, 0].set_ylabel('Z-Score')

# Highlight the outliers based on the z-score
outliers = df[df['outlier'] == True]
axes[4, 0].scatter(outliers.index, outliers['z_score'], color='red', label='Outliers', marker='x')
axes[4, 0].legend()

# Hide the last empty subplot
axes[4, 1].axis('off')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
sns.set(style="whitegrid")

# Plot SMA and other features
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
fig.suptitle('Maneuver Detection Visualizations', fontsize=16)

# Scatter plot for SMA values over time with color coding based on 'outlier' feature
axes[0, 0].scatter(df.index, df['SMA'], c=df['outlier'].map({True: 'red', False: 'blue'}), label='SMA', alpha=0.7)
axes[0, 0].set_title('SMA Over Time')
axes[0, 0].set_ylabel('SMA')
axes[0, 0].legend(['Outliers', 'SMA'])

# Scatter plot for SMA velocity (first derivative) with color coding
axes[0, 1].scatter(df.index, df['sma_velocity'], c=df['sma_velocity'], cmap='coolwarm', label='Velocity', alpha=0.7)
axes[0, 1].set_title('SMA Velocity (First Derivative)')
axes[0, 1].set_ylabel('Velocity')

# Scatter plot for SMA acceleration (second derivative) with color coding
axes[1, 0].scatter(df.index, df['sma_acceleration'], c=df['sma_acceleration'], cmap='viridis', label='Acceleration', alpha=0.7)
axes[1, 0].set_title('SMA Acceleration (Second Derivative)')
axes[1, 0].set_ylabel('Acceleration')

# Scatter plot for rolling mean of SMA with color coding
axes[1, 1].scatter(df.index, df['rolling_mean'], c=df['rolling_mean'], cmap='plasma', label='Rolling Mean', alpha=0.7)
axes[1, 1].set_title('Rolling Mean of SMA')
axes[1, 1].set_ylabel('Rolling Mean')

# Scatter plot for rolling standard deviation of SMA with color coding
axes[2, 0].scatter(df.index, df['rolling_std'], c=df['rolling_std'], cmap='cool', label='Rolling Std Dev', alpha=0.7)
axes[2, 0].set_title('Rolling Std Dev of SMA')
axes[2, 0].set_ylabel('Rolling Std Dev')

# Scatter plot for rolling window SMA differences with color coding
axes[2, 1].scatter(df.index, df['rolling_diff'], c=df['rolling_diff'], cmap='RdYlBu', label='Rolling Difference', alpha=0.7)
axes[2, 1].set_title('Rolling Window SMA Differences')
axes[2, 1].set_ylabel('Rolling Difference')

# Scatter plot for exponential moving average (EMA) of SMA with color coding
axes[3, 0].scatter(df.index, df['ema'], c=df['ema'], cmap='Blues', label='EMA', alpha=0.7)
axes[3, 0].set_title('Exponential Moving Average of SMA')
axes[3, 0].set_ylabel('EMA')

# Scatter plot for cumulative sum of SMA with color coding
axes[3, 1].scatter(df.index, df['cumulative_sum'], c=df['cumulative_sum'], cmap='Purples', label='Cumulative Sum', alpha=0.7)
axes[3, 1].set_title('Cumulative Sum of SMA')
axes[3, 1].set_ylabel('Cumulative Sum')

# Scatter plot for z-score with color coding for anomalies
axes[4, 0].scatter(df.index, df['z_score'], c=df['outlier'].map({True: 'red', False: 'grey'}), alpha=0.7, label='Z-Score')
axes[4, 0].set_title('Z-Score for Anomaly Detection')
axes[4, 0].set_ylabel('Z-Score')
axes[4, 0].legend(['Outliers', 'Z-Score'])

# Hide the last empty subplot
axes[4, 1].axis('off')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


**Data Preprocessing**


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore

# Load the dataset
# Replace 'dataset.csv' with your actual dataset file path
df = pd.read_csv('SMA_data.csv', parse_dates=['Datetime'])

# 1. Handling Missing Data
# Interpolating missing values
df['SMA'] = df['SMA'].interpolate(method='linear')

# 2. Time-Based Feature Engineering
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Hour'] = df['Datetime'].dt.hour

# Create time differences in seconds
df['Time_Diff'] = df['Datetime'].diff().dt.total_seconds()

# 3. SMA-Based Feature Engineering
df['SMA_Diff'] = df['SMA'].diff()  # Difference between consecutive SMA values
df['SMA_Pct_Change'] = df['SMA'].pct_change()  # Percentage change
df['SMA_CumSum'] = df['SMA_Diff'].cumsum()  # Cumulative sum of SMA differences

# 4. Data Normalization/Scaling
# Standardization (Z-score normalization)
scaler_standard = StandardScaler()
df['SMA_Standardized'] = scaler_standard.fit_transform(df[['SMA']])

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df['SMA_MinMax'] = scaler_minmax.fit_transform(df[['SMA']])

# 5. Handling Time Gaps
# Resampling to a regular interval (e.g., hourly)
df_resampled = df.set_index('Datetime').resample('1H').mean().interpolate()  # Resample to hourly and interpolate missing values

# Alternatively, flag irregular time gaps without resampling
df['Irregular_Time_Gap'] = df['Time_Diff'] > (df['Time_Diff'].mean() + 3 * df['Time_Diff'].std())

# 6. Outlier Detection and Removal
# Z-score based outlier detection
df['SMA_Zscore'] = zscore(df['SMA'])
df['Outlier_Flag'] = np.abs(df['SMA_Zscore']) > 3  # Flag as outlier if Z-score > 3

# Outlier Treatment (e.g., capping)
df.loc[df['Outlier_Flag'], 'SMA'] = df['SMA'].median()

# 7. Datetime Indexing
df.set_index('Datetime', inplace=True)

# 8. Differencing for Stationarity
df['SMA_Diff_1st'] = df['SMA'].diff()  # First-order differencing
df['SMA_Diff_Seasonal'] = df['SMA'].diff(24)  # Seasonal differencing (e.g., daily if hourly data)

# 9. Lag Features
# Creating lagged variables
df['SMA_Lag_1'] = df['SMA'].shift(1)
df['SMA_Lag_2'] = df['SMA'].shift(2)
df['SMA_Lag_3'] = df['SMA'].shift(3)

# Display the processed data
print(df.head())

# Save the preprocessed dataset to a new CSV file
df.to_csv('preprocessed_dataset.csv', index=True)

**Feature Extraction**

df = pd.read_csv('SMA_data.csv')
df.info()
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import PolynomialFeatures



df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Rolling Statistics (Window-Based Features)
df['rolling_mean'] = df['SMA'].rolling(window=3).mean()
df['rolling_std'] = df['SMA'].rolling(window=3).std()
df['rolling_min'] = df['SMA'].rolling(window=3).min()
df['rolling_max'] = df['SMA'].rolling(window=3).max()

# Linear Trend Feature
X = np.arange(len(df)).reshape(-1, 1)
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)
df['linear_trend'] = X_poly[:, 1]

# Exponential Moving Average
df['ema'] = df['SMA'].ewm(span=5, adjust=False).mean()

# Frequency Domain Features (Fourier Transform)
fft_values = fft(df['SMA'].values)
df['fft_real'] = np.real(fft_values)
df['fft_imag'] = np.imag(fft_values)

# Lag-Based Features
df['lag_1'] = df['SMA'].shift(1)
df['lag_2'] = df['SMA'].shift(2)

# Autoregressive Features
model = AutoReg(df['SMA'].dropna(), lags=2)
ar_model = model.fit()
df['ar_coef_1'] = ar_model.params[1]  # First AR coefficient
df['ar_coef_2'] = ar_model.params[2]  # Second AR coefficient

# Cumulative Features
df['cumulative_sum'] = df['SMA'].cumsum()
df['cumulative_max'] = df['SMA'].cummax()
df['cumulative_min'] = df['SMA'].cummin()

# Differencing and Change Features
df['diff_1'] = df['SMA'].diff(1)
df['diff_2'] = df['SMA'].diff(2)
df['pct_change'] = df['SMA'].pct_change()
df['direction'] = np.sign(df['SMA'].diff())

# Anomaly/Change Detection Features
df['z_score'] = (df['SMA'] - df['SMA'].mean()) / df['SMA'].std()
df['outlier'] = np.abs(df['z_score']) > 2  # Mark points as outliers if Z-score > 2

# Time-Based Features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Segment-Level Features (Segment by day)
df['date'] = df.index.date
daily_groups = df.groupby('date')['SMA']
df['daily_mean'] = daily_groups.transform('mean')
df['daily_std'] = daily_groups.transform('std')

# Drop temporary columns used for segmentation
df.drop(columns=['date'], inplace=True)

# Display the resulting DataFrame with extracted features
print(df.head())


