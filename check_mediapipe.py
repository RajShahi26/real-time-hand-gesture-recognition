import mediapipe as mp

print("MediaPipe module:", mp)
print("Location:", mp.__file__)
print("Version:", getattr(mp, "__version__", "No version found"))

print("\nContains solutions?")
print(hasattr(mp, "solutions"))

print("\nAvailable attributes:")
print(dir(mp)[:30])