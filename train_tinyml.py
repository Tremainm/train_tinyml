import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import pickle

# ── Load & normalise ──────────────────────────────────────────────────────────
df = pd.read_csv("dht_readings.csv", names=["temperature", "humidity"])
data = df[["temperature", "humidity"]].values  # shape: (N, 2)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)  # scale both features to [0, 1]

# Save the scaler — you'll need its min/scale values baked into the firmware
print("Scaler min_:", scaler.min_)
print("Scaler scale_:", scaler.scale_)

# ── Build autoencoder ─────────────────────────────────────────────────────────
input_layer = keras.Input(shape=(2,))
encoded = keras.layers.Dense(4, activation="relu")(input_layer)   # bottleneck
decoded = keras.layers.Dense(2, activation="sigmoid")(encoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
autoencoder.fit(
    data_scaled, data_scaled,   # input == target for an autoencoder
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    shuffle=True
)

# ── Compute threshold ─────────────────────────────────────────────────────────
reconstructions = autoencoder.predict(data_scaled)
reconstruction_errors = np.mean(np.square(data_scaled - reconstructions), axis=1)
threshold = float(np.percentile(reconstruction_errors, 95))
print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")
# Hardcode this value into your firmware

# ── Export to TFLite (int8 quantised) ─────────────────────────────────────────
def representative_dataset():
    for sample in data_scaled:
        yield [sample.reshape(1, 2).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("anomaly_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model)} bytes")