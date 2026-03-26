import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("labelled_readings.csv", names=["temperature", "humidity", "label"])

print(df.groupby("label")[["temperature", "humidity"]].describe())

# Scatter plot — if the three classes overlap heavily, the model will struggle
colours = {"NORMAL": "blue", "WINDOW_OPEN": "green", "HEATING_ON": "red"}
for label, group in df.groupby("label"):
    plt.scatter(group["temperature"], group["humidity"],
                label=label, alpha=0.4, c=colours[label])

plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.legend()
plt.title("Class separation")
plt.savefig("class_separation.png")
print("Saved class_separation.png")