# Import required libraries
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import joblib
# Define the dataset path
dataset_path = "C:\\Users\\Ajay Kumar\\OneDrive\\Desktop\\capstone project\\Eye_diseases"  # Replace with the folder path

# Initialize lists to store images and labels
images = []
labels = []

# Loop through each class folder
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):  # Check if it's a folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            # Skip non-image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")  # Convert to RGB
                image = image.resize((128, 128))  # Resize to 128x128
                images.append(np.array(image))
                labels.append(class_folder)  # Use folder name as the label
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

print(f"Loaded {len(X)} images with {len(np.unique(y))} classes.")
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize images
X = X / 255.0  # Scale pixel values to [0, 1]

# Flatten images for Random Forest model
X_flat = X.reshape(X.shape[0], -1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "eye_disease_model.pkl")
print("Model saved as eye_disease_model.pkl")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
def predict_disease_or_healthy(image_path, model, label_encoder, threshold=0.5):
    """
    Predict the disease for a given eye image or classify it as 'Healthy Eyes' if confidence is low.

    :param image_path: Path to the input image.
    :param model: Trained Random Forest model.
    :param label_encoder: Fitted LabelEncoder for decoding class labels.
    :param threshold: Confidence threshold below which the result is 'Healthy Eyes'.
    :return: Predicted disease or 'Healthy Eyes'.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))  # Resize to match training data
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_flattened = image_array.flatten().reshape(1, -1)  # Flatten the image

        # Predict probabilities
        probabilities = model.predict_proba(image_flattened)[0]
        max_confidence = max(probabilities)
        predicted_class = np.argmax(probabilities)

        if max_confidence < threshold:
            return "Healthy Eyes"  # If confidence is below threshold, classify as healthy
        
        # Decode the class label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        return predicted_label

    except Exception as e:
        return f"Error processing image: {e}"
input_image_path = "C:\\Users\\Ajay Kumar\\OneDrive\\Desktop\\capstone project\\Eye_diseases\\Cataracts\\image-2.jpeg"  # Replace with the actual path

# Predict with a confidence threshold
predicted_disease = predict_disease_or_healthy(input_image_path, model, label_encoder, threshold=0.5)
print(f"The prediction is: {predicted_disease}")
