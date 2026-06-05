from tensorflow.keras.models import load_model

model = load_model("gesture_model.keras")

print("Model loaded successfully!")
print("Input Shape :", model.input_shape)
print("Output Shape:", model.output_shape)

