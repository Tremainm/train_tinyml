import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("labelled_dummy_readings.csv", names=["temperature", "humidity", "label"])

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(df["label"])
print("Class order (note this down for firmware):", list(le.classes_))
# LabelEncoder sorts alphabetically, so will be:
# 0 = HEATING_ON, 1 = NORMAL, 2 = WINDOW_OPEN

X = df[["temperature", "humidity"]].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) # normalised to a range [0, 1]

print("Scaler min_:", scaler.min_)
print("Scaler scale_:", scaler.scale_)

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y ensures each split has proportional class representation

n_classes = len(le.classes_)  # 3

# 2 layer network: input - temp + humid
#                  8 neurons, ReLu activation (model can learn patterns), Every input is connected to each 8 neurons.
#                  2 * 8 + 8 = 24 learnable weights
#                  outputs - 3 neurons (Softmax), prob per class that sum to 1
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(n_classes, activation="softmax")
])

# Adam (Adaptive Movement Estimation): updates model's weights after each batch. Auto adjusts learning rate per param.
# loss: How wrong the predictions are. 
#       "Sparse"                 - plain ints (0, 1, 2). 
#       categorical_crossenrtopy - heavily penalises confident wrong predictions. Pushes model to assign high prob to correct class.
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train,
          epochs=100,
          batch_size=16,
          validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc:.2%}")

# Export to TFLite int8
def representative_dataset():
    for sample in X_scaled:
        yield [sample.reshape(1, 2).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# compatibility workaround for microcontroller runtimes that don't support per-channel quantization
converter._experimental_disable_per_channel_quantization_for_dense_layers = True

tflite_model = converter.convert()

with open("context_dummy_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model)} bytes")