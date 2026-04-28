# TinyML Sensor Models

This project trains a TensorFlow Lite model to classify environmental context from DHT22 sensor readings (temperature + humidity), for deployment on an ESP32-C3 Matter sensor node.

The original approach was anomaly detection using an autoencoder (`train_tinyml.py`). This was abandoned in favour of context classification, which provides more actionable and interpretable output for the firmware. **Context classification is the active pipeline.**

The trained model is deployed to the ESP32-C3 DHT22 Matter Sensor firmware here: [https://github.com/Tremainm/ESP32C3-DHT22-Sensor](https://github.com/Tremainm/ESP32C3-DHT22-Sensor)

---

## Context Classification Pipeline

### Step 1 — Generate synthetic training data (`synthetic_data.py`)

Real DHT22 readings don't provide enough labelled variety to cover all three target conditions, so synthetic data is generated using Gaussian distributions tuned to match physically realistic sensor behaviour:

| Class | Temperature | Humidity | Meaning |
|---|---|---|---|
| `HEATING_ON` | ~27°C ± 1.0 | ~45% ± 2.0 | Radiator nearby, air drying out |
| `NORMAL` | ~21°C ± 0.6 | ~55% ± 2.0 | Typical indoor conditions |
| `WINDOW_OPEN` | ~15°C ± 1.0 | ~73% ± 2.5 | Cold damp outside air entering |

500 readings per class (1500 total) are generated and saved to `labelled_dummy_readings.csv`. Values are rounded to 1 decimal place to match real DHT22 output resolution.

### Step 2 — Verify class separation (`data_verify.py`)

Plots a scatter graph of all three classes (`class_separation_dummy.png`) and prints per-class statistics. If the clusters overlap heavily the model will struggle — this is a sanity check before committing to training.

### Step 3 — Train & export (`train_tinyml_context.py`)

**Data loading & label encoding**

The CSV is loaded and string labels are encoded to integers using `LabelEncoder`, which sorts alphabetically:
- `0 = HEATING_ON`
- `1 = NORMAL`
- `2 = WINDOW_OPEN`

This order must be hardcoded in the firmware to interpret model output correctly.

**Feature scaling**

`MinMaxScaler` scales both temperature and humidity to `[0, 1]`. The resulting `min_` and `scale_` values are printed to the console — these must be baked into the firmware to apply the same normalisation to live sensor readings at inference time.

**Train/test split**

80% training / 20% test with `stratify=y` to ensure proportional class representation in both sets.

**Model architecture**

A minimal two-layer neural network:

```
Input (2)  →  Dense 8, ReLU  →  Dense 3, Softmax  →  Output (3 probabilities)
```

- Input: scaled temperature and humidity
- Hidden layer: 8 neurons with ReLU activation to learn non-linear boundaries between classes
- Output: 3 neurons with Softmax, giving a probability per class — the class with the highest probability is the prediction
- Total parameters: 51 — intentionally tiny to fit in microcontroller flash

**Training**

- Optimizer: Adam
- Loss: `sparse_categorical_crossentropy` (labels are integers, not one-hot vectors)
- 100 epochs, batch size 16

**int8 Quantisation & export**

The trained float32 model is converted to a fully quantised int8 TFLite model:

- A `representative_dataset` generator feeds real training samples to the converter so it can calibrate the activation value ranges across every layer (full integer quantisation, not just weight quantisation)
- All ops are forced to int8 via `TFLITE_BUILTINS_INT8`
- Both `inference_input_type` and `inference_output_type` are set to `tf.int8` — the model's I/O boundary is int8, not float
- Per-channel quantisation is disabled on Dense layers (`_experimental_disable_per_channel_quantization_for_dense_layers = True`) for compatibility with the TFLite Micro runtime on the ESP32-C3

Output: `context_dummy_model.tflite`

---

## Firmware Integration

Three things must be embedded in the ESP32-C3 firmware:

### 1. MinMaxScaler parameters
Printed when the training script runs. Applied to raw sensor readings before inference:
```c
float scaled = (raw_value - min_) * scale_;
```

### 2. Input quantisation (float → int8)
The model expects int8 input. After scaling:
```c
int8_t quantised = (int8_t)(scaled / input_scale + input_zero_point);
```
`input_scale` and `input_zero_point` are read from the TFLite flatbuffer at runtime via the interpreter API.

### 3. Output interpretation
Find the argmax of the three int8 output values — the index is the predicted class:
- `0 = HEATING_ON`
- `1 = NORMAL`
- `2 = WINDOW_OPEN`

---

## Files

| File | Description |
|---|---|
| `train_tinyml_context.py` | Context classifier — train & export (active pipeline) |
| `synthetic_data.py` | Generates `labelled_dummy_readings.csv` |
| `data_verify.py` | Plots class separation scatter graph |
| `labelled_dummy_readings.csv` | Synthetic labelled readings (1500 rows) |
| `labelled_readings.csv` | Real labelled DHT22 readings |
| `context_dummy_model.tflite` | Exported context model (synthetic data) |
| `context_model.tflite` | Exported context model (real data) |
| `context_dummy_model_data.h` / `context_model_data.h` | C-style array of model bytes for embedding in firmware |
| `class_separation_dummy.png` | Scatter plot of synthetic class distributions |
| `train_tinyml.py` | Anomaly detection autoencoder (abandoned, kept for reference) |
| `anomaly_model.tflite` | Exported anomaly model (not used) |
