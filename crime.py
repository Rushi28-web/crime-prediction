import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

np.random.seed(42)

# Generate synthetic data function
def generate_crime_data(base, step, noise):
    return [base + i * step + np.random.randint(-noise, noise + 1) for i in range(36)]

# City configuration
cities_config = {
    'Ballari':     {'base': 100, 'step': 2, 'noise': 5},
    'Bengaluru':   {'base': 320, 'step': 3, 'noise': 8},
    'Mysuru':      {'base': 150, 'step': 1, 'noise': 4},
    'Hubballi':    {'base': 180, 'step': 1.5, 'noise': 6},
    'Mangaluru':   {'base': 140, 'step': 2.2, 'noise': 5},
    'Kalaburagi':  {'base': 160, 'step': 1.8, 'noise': 7},
}

forecast_sums = {}
growth_df = []
monthly_forecasts = {}

# Forecast & calculate growth
for city, config in cities_config.items():
    crimes = generate_crime_data(config['base'], config['step'], config['noise'])
    ts = pd.Series(crimes, index=pd.RangeIndex(start=1, stop=37))
    model = auto_arima(ts, seasonal=False, suppress_warnings=True, stepwise=True)
    forecast = model.predict(n_periods=12)

    forecast_sums[city] = np.sum(forecast)
    monthly_forecasts[city] = forecast

    last_actual = np.sum(ts[-12:])
    predicted = np.sum(forecast)
    growth = ((predicted - last_actual) / last_actual) * 100
    growth_df.append((city, round(growth, 2)))

# DataFrames
forecast_df = pd.DataFrame(list(forecast_sums.items()), columns=['City', 'Total Predicted Crimes'])
growth_table = pd.DataFrame(growth_df, columns=["City", "Predicted Growth %"])

# ----- Combined Subplots -----
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Pie Chart
axs[0, 0].pie(forecast_df['Total Predicted Crimes'], labels=forecast_df['City'], autopct='%1.1f%%', startangle=140)
axs[0, 0].set_title("Pie Chart")

# Bar Chart
axs[0, 1].bar(forecast_df['City'], forecast_df['Total Predicted Crimes'], color='skyblue')
axs[0, 1].set_title("Bar Chart")

# Histogram
axs[1, 0].hist(forecast_df['Total Predicted Crimes'], bins=6, color='lightcoral', edgecolor='black')
axs[1, 0].set_title("Histogram")

# Line Plot
axs[1, 1].plot(forecast_df['City'], forecast_df['Total Predicted Crimes'], marker='o', color='purple')
axs[1, 1].set_title("Line Plot")

for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.4)

fig.suptitle("Predicted Crime Comparison Across Cities", fontsize=16)
plt.tight_layout()
plt.show()

# ----- Monthly Trends for All Cities -----
plt.figure(figsize=(12, 6))
for city, forecast in monthly_forecasts.items():
    months = list(range(37, 49))
    plt.plot(months, forecast, label=city, marker='o')
plt.title("Monthly Predicted Crimes (Next 12 Months)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Predicted Crimes")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ----- Tables -----
print("\n=== Summary Table: Total Predicted Crimes (Next 12 Months) ===")
print(forecast_df.sort_values(by='Total Predicted Crimes', ascending=False).to_string(index=False))

print("\n=== Predicted Growth in Crime (Next 12 Months vs Last 12 Months) ===")
print(growth_table.sort_values(by="Predicted Growth %", ascending=False).to_string(index=False))

# ----- Export to Excel -----
forecast_df.to_excel("predicted_crimes.xlsx", index=False)
growth_table.to_excel("predicted_growth.xlsx", index=False)
print("\nData exported to 'predicted_crimes.xlsx' and 'predicted_growth.xlsx'")
