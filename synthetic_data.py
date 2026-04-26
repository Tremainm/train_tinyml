import numpy as np
import pandas as pd

np.random.seed(42)
N = 500  # readings per class

# ── NORMAL ────────────────────────────────────────────────────────────────────
# Typical indoor room: 20-22°C, 53-63% humidity
normal_temp = np.random.normal(loc=21.0, scale=0.6, size=N)
normal_hum  = np.random.normal(loc=55.0, scale=2.0, size=N)

# ── HEATING_ON ────────────────────────────────────────────────────────────────
# Radiator nearby: temperature clearly elevated, humidity drops as air dries
heating_temp = np.random.normal(loc=27.0, scale=1.0, size=N)
heating_hum  = np.random.normal(loc=45.0, scale=2.0, size=N)

# ── WINDOW_OPEN ───────────────────────────────────────────────────────────────
# Cold outside air: temperature drops, humidity rises (damp outside air)
window_temp = np.random.normal(loc=15.0, scale=1.0, size=N)
window_hum  = np.random.normal(loc=73.0, scale=2.5, size=N)

# ── Combine and label ─────────────────────────────────────────────────────────
rows = (
    [(t, h, "NORMAL")      for t, h in zip(normal_temp,  normal_hum)]  +
    [(t, h, "HEATING_ON")  for t, h in zip(heating_temp, heating_hum)] +
    [(t, h, "WINDOW_OPEN") for t, h in zip(window_temp,  window_hum)]
)

df = pd.DataFrame(rows, columns=["temperature", "humidity", "label"])

# Round to 1 decimal place to match real DHT22 output resolution
df["temperature"] = df["temperature"].round(1)
df["humidity"]    = df["humidity"].round(1)

# Shuffle so classes aren't in blocks
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save without header to match your existing capture script format
df.to_csv("labelled_dummy_readings.csv", index=False, header=False)

print(f"Generated {len(df)} rows")
print(df.groupby("label")[["temperature", "humidity"]].describe().round(2))