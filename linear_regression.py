import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define a simple Sequential model
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500, verbose=0)

# Predict
prediction = model.predict(np.array([[10.0]]))
print(f"Prediction for x=10: {prediction[0][0]}")

# Show learned parameters
weights, biases = model.layers[0].get_weights()
print(f"Learned weight (slope): {weights[0][0]}")
print(f"Learned bias (intercept): {biases[0]}")
