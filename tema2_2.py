import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Function to load train images from a folder and assign labels
def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                            images.append(img)
                            labels.append(subfolder)
                            print('Labels \n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return images, labels

# Function to load test images from a folder and assign labels
def load_test_images_from_folder(folder, target_shape=None):
    test_images = []
    test_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                            test_images.append(img)
                            test_labels.append(subfolder)
                            print('Labels \n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return test_images, test_labels

def load_val_images_from_folder(folder, target_shape=None):
    val_images = []
    val_labels = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))

                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                            val_images.append(img)
                            val_labels.append(subfolder)
                            print('Labels \n', labels)
                        else:
                            print(f"Warning: Unable to load {filename}")

    return val_images, val_labels

# Function to load images and labels from the dataset
def load_dataset(root_folder):
    images = []
    labels = []
    class_mapping = {} # To map class names to numeric labels
    for class_label, class_name in enumerate(os.listdir(root_folder)): 
        class_mapping[class_label] = class_name
        class_folder = os.path.join(root_folder, class_name)

        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = imread(image_path)


            if image.ndim == 3:
                image = np.mean(image, axis=2) # Convert RGB image to grayscale
                

            image = resize(image, (64, 64), anti_aliasing=True) # Adjust image size as needed 

            


            images.append(image.flatten()) # Flatten the image
            labels.append(class_label)
    return np.array(images), np.array(labels), class_mapping


# Function to save results to a CSV file
def save_results(algorithm, y_true, y_pred, class_mapping):
    results_df = pd.DataFrame({'True Label': [class_mapping[label] for label in y_true],
    'Predicted Label': [class_mapping[label] for label in y_pred]})
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    results_file = os.path.join(results_folder, f'{algorithm}_results.csv') 
    results_df.to_csv(results_file, index=False)
    print(f'Results saved for {algorithm} at {results_file}')


# Function to save test images with predictions
def save_test_images(algorithm, X_test, y_true, y_pred, class_mapping, probabilities): 
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    for idx, (true_label, pred_label, prob_dist) in enumerate(zip(y_true, y_pred, probabilities)):
        image = X_test[idx].reshape(64, 64, 1) # Reshape flattened image
        plt.imshow(image)
        plt.title(f'True Label: {class_mapping[true_label]}\nPredicted Label: {class_mapping[pred_label]}') 
        plt.savefig(os.path.join(results_folder, f'{algorithm}_test_image_{idx}.png'))
        plt.close()
        # Plot probability distribution
        plt.bar(class_mapping.values(), prob_dist, color='blue')
        plt.title(f'Probability Distribution - Test Image {idx}')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_probability_distribution_{idx}.png')) 
        plt.close()

    print(f'Test images saved for {algorithm} in {results_folder}')

# Folder paths
data_folder = './train1'  # Folder with training data
test_folder = './test'  # Folder with test data
val_folder = './val'  # Folder with validation data

# Load images and labels from the 'dataset' folder and resize them to (200, 200)
images, labels = load_train_images_from_folder(data_folder, target_shape=(250, 250))

# Load validation images and labels from the 'val' folder
val_images, val_labels = load_val_images_from_folder(val_folder, target_shape=(250, 250))

# Combine training and validation data
images += val_images
labels += val_labels

# Convert labels to binary (0 or 1)
labels_binary = [1 if label == 'NORMAL' else 0 for label in labels]

# Reshape the images and convert them to grayscale
image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]

# Convert the list of 1D arrays to a 2D numpy array
image_data = np.array(image_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)

# Load TEST images from the 'test' folder
test_images, test_labels = load_test_images_from_folder(test_folder, target_shape=(250, 250))

# Convert labels to binary (0 or 1), reshape and convert to grayscale and scale dthe TEST data
test_labels_binary = [1 if label == 'NORMAL' else 0 for label in test_labels]
test_image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in test_images]
test_image_data = np.array(test_image_data)
scaled_test_data = scaler.transform(test_image_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(scaled_data, labels_binary, test_size=0.2, random_state=42)

# Train a Random Forest model
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Predictions on the validation set
validation_predictions = random_forest_model.predict(X_val)

# Predictions on the train set
train_predictions = random_forest_model.predict(X_train)
### RANDOM FOREST###

# Evaluate performance on the validation set
accuracy_train_rf = accuracy_score(y_train, train_predictions)
precision_val = precision_score(y_val, validation_predictions)
recall_val = recall_score(y_val, validation_predictions, zero_division=1)
f1_val = f1_score(y_val, validation_predictions)
confusion_matrix_val = confusion_matrix(y_val, validation_predictions)

print("Performance on Validation Set:")
print(f"Accuracy: {accuracy_train_rf:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_val)

# Predictions on the test set
test_predictions = random_forest_model.predict(scaled_test_data)

# Evaluate performance on the test set
accuracy_test_rf = accuracy_score(test_labels_binary, test_predictions)
precision_test = precision_score(test_labels_binary, test_predictions)
recall_test = recall_score(test_labels_binary, test_predictions, zero_division=1)
f1_test = f1_score(test_labels_binary, test_predictions)
confusion_matrix_test = confusion_matrix(test_labels_binary, test_predictions)

print("\nPerformance on Test Set:")
print(f"Accuracy: {accuracy_test_rf:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_test)

#----------------------------------------

# Load your dataset
dataset_folder = "./data" # Change this to the path of your dataset folder 
X, y, class_mapping = load_dataset(dataset_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example 1: k-Nearest Neighbors (k-NN)
# Initialize k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_classifier.fit(X_train, y_train) # Make predictions on the test set
y_test_pred_knn = knn_classifier.predict(X_test)
y_train_pred_knn = knn_classifier.predict(X_train)
# Save results and test images for k-NN
save_results('knn', y_test, y_test_pred_knn, class_mapping)
probabilities_knn = knn_classifier.predict_proba(X_test)
save_test_images('knn', X_test, y_test, y_test_pred_knn, class_mapping, probabilities_knn)

# Example 2: Naive Bayes (Gaussian Naive Bayes) # Initialize Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred_nb = nb_classifier.predict(X_test)

# Save results and test images for Naive Bayes
save_results('naive_bayes', y_test, y_test_pred_nb, class_mapping)
probabilities_nb = nb_classifier.predict_proba(X_test)
save_test_images('naive_bayes', X_test, y_test, y_test_pred_nb, class_mapping, probabilities_nb)

# EVALUAREA REZULTATELOR
# Evaluate and print accuracy for k-NN
accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
print(f'Acuratețe test pentru k-NN: {accuracy_knn:.4f}')
accuracy_train_knn = accuracy_score(y_train, y_train_pred_knn)
print(f'Acuratețea train pentru k-NN pe setul de antrenare: {accuracy_train_knn}')


# Evaluate and print accuracy for Naive Bayes
accuracy_nb = accuracy_score(y_test, y_test_pred_nb)
print(f'Acuratețe pentru Naive Bayes: {accuracy_nb:.4f}')
y_train_pred_nb = nb_classifier.predict(X_train)
accuracy_train_nb = accuracy_score(y_train, y_train_pred_nb)
print(f'Acuratețea pentru Gaussian Naive Bayes pe setul de antrenare: {accuracy_train_nb}')

# Classification report for k-NN
print("Raport de clasificare pentru k-NN:")
print(classification_report(y_test, y_test_pred_knn, target_names=class_mapping.values()))

# Classification report for Naive Bayes
print("Raport de clasificare pentru Naive Bayes:")
print(classification_report(y_test, y_test_pred_nb, target_names=class_mapping.values()))


labels = ['rdm-F Train','rdm-F Test' ,'k-NN Train', 'k-NN Test', 'Naive Bayes Train', 'Naive Bayes Test']
accuracies = [accuracy_train_rf,accuracy_test_rf,accuracy_train_knn, accuracy_knn, accuracy_train_nb, accuracy_nb]

plt.bar(labels, accuracies, color=['blue', 'orange', 'green', 'red', 'yellow', 'purple'])
plt.ylabel('Accuracy')
plt.title('Acuratețea pentru Modelele Random Forest, k-NN și Naive Bayes')
plt.show()

# Etichetele claselor pentru afișare pe axa x
clasificatori = ['k-NN', 'Naive Bayes']
# Valori acuratețe
valori_acuratete = [accuracy_knn , accuracy_nb]

# Creare diagramă de bare
plt.bar(clasificatori, valori_acuratete, color=['blue', 'green'])
plt.ylim(0, 1)  

# Adăugare etichete și titluri
plt.xlabel('Clasificatori')
plt.ylabel('Acuratețe')
plt.title('Acuratețe pentru k-NN și Naive Bayes')

# Afișare diagramă
plt.show()

# Pentru k-NN
conf_matrix_knn = confusion_matrix(y_test, y_test_pred_knn)
print("Matricea de confuzie pentru k-NN:\n", conf_matrix_knn)

# Pentru Naive Bayes
conf_matrix_nb = confusion_matrix(y_test, y_test_pred_nb)
print("Matricea de confuzie pentru Naive Bayes:\n", conf_matrix_nb)

# Plot confusion matrix for k-NN
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_mapping.values(), yticklabels=class_mapping.values())
plt.title('Confusion Matrix for k-NN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot confusion matrix for Naive Bayes
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_mapping.values(), yticklabels=class_mapping.values())
plt.title('Confusion Matrix for Naive Bayes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


