import pandas as pd
import os

# telling the script where it actually lives so it doesn't get lost
script_dir = os.path.dirname(os.path.abspath(__file__))
weather_path = os.path.join(script_dir, 'raw', 'Weather.csv')
elec_path = os.path.join(script_dir, 'raw', 'Electricity.csv')
pop_path = os.path.join(script_dir, 'raw', 'Population.csv')

# 1. trim the weather data because it's too chunky
df_weather = pd.read_csv(weather_path, low_memory=False) 
# dumped the boring stuff: max_temp, min_temp, pressure, and clouds
weather_cols = [
    'date', 'avg_temperature', 'avg_relative_humidity', 
    'avg_wind_speed', 'avg_hourly_health_index', 'rain', 'snow'
]
df_weather = df_weather[weather_cols]
df_weather['date'] = pd.to_datetime(df_weather['date'])

# 2. figure out the electricity vibes
df_elec = pd.read_csv(elec_path)
df_elec['date'] = pd.to_datetime(df_elec['date'])
# smashing hourly data into daily averages
df_elec_daily = df_elec.resample('D', on='date')['hourly_demand'].mean().reset_index()
df_elec_daily.rename(columns={'hourly_demand': 'avg_daily_demand'}, inplace=True)

# 3. sorting out the population numbers
df_pop = pd.read_csv(pop_path)
df_pop = df_pop[['year', 'month', 'population_growth']]
df_pop['year'] = pd.to_numeric(df_pop['year'], errors='coerce')
df_pop['month'] = pd.to_numeric(df_pop['month'], errors='coerce')
# getting rid of the % sign because math hates it
df_pop['population_growth'] = df_pop['population_growth'].astype(str).str.replace('%', '', regex=False).astype(float)

# 4. making weather and electricity finally meet
df_combined = pd.merge(df_weather, df_elec_daily, on='date', how='inner')

# 5. sticking population in there too
df_combined['year'] = df_combined['date'].dt.year
df_combined['month'] = df_combined['date'].dt.month
df_combined = pd.merge(df_combined, df_pop, on=['year', 'month'], how='left')
# fill in the blanks so the code doesn't freak out
df_combined['population_growth'] = df_combined['population_growth'].ffill().fillna(0)

# 6. only keeping the years we actually care about (2002-2025)
df_combined = df_combined[(df_combined['year'] >= 2002) & (df_combined['year'] <= 2025)]
df_combined = df_combined.sort_values('date').reset_index(drop=True)

# 7. making features that a model can actually understand
# used numbers for days because machines don't speak "tuesday"
df_combined['day_of_week'] = df_combined['date'].dt.dayofweek 
df_combined['is_weekend'] = df_combined['day_of_week'].isin([5, 6]).astype(int)

# spitting out the final file
df_combined.to_csv('daily_weather_and_demand_2002_2025.csv', index=False)

print("slimmed data process complete!")
print(f"features kept: {df_combined.columns.tolist()}")
print(f"total rows: {len(df_combined)}")