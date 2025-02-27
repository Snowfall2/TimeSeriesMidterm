import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load image and convert to HSV
def load_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    color_channel = hsv_image[:, :, :] 
    return color_channel

# Convert to histogram
def extract_hsv_histogram(image, bins=90):
    hue_hist = cv.calcHist([image], [0], None, [bins], [0, 180]).flatten()
    sat_hist = cv.calcHist([image], [1], None, [255], [0, 255]).flatten()
    val_hist = cv.calcHist([image], [2], None, [255], [0, 255]).flatten()
    
    # # Normalize histograms to 1 for each component
    hue_hist /= np.sum(hue_hist)
    sat_hist /= np.sum(sat_hist)
    val_hist /= np.sum(val_hist)
    return np.concatenate([val_hist, sat_hist])

# Load dataset of images 
image_paths = {
    "surrealism": "artbench-10-imagefolder-split/train/surrealism",
    "realism": "artbench-10-imagefolder-split/train/realism"
}

X, y = [], []
for style, path in image_paths.items():
    for image in os.listdir(path):
        try:
            image = load_image(path + "/" + image)
            file_name = image
            features = extract_hsv_histogram(image)
            X.append(features)
            y.append(style)
        except Exception:
            pass

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



for i in range(1,100,2):
    knnPredict = KNeighborsClassifier(n_neighbors=i)
    knnPredict.fit(X_train, y_train)
    print("------")
    pred = knnPredict.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("k = {}, Accuracy = {}".format(i, acc))
    
# # Test on a new image
# new_image = load_image("{}/max-walter-svanberg_portratt-av-en-stjarna-iii.jpg".format(image_paths["surrealism"]))
# new_image = load_image("{}/Arkadiusz_Dzielawski_Pegasus.jpg".format(image_paths["surrealism"]))
# new_image = load_image("{}/craig-mullins_untitled-14.jpg".format(image_paths["realism"]))
new_image = load_image("{}/dobri-dobrev_marketplace-1932.jpg".format(image_paths["realism"]))
new_features = [extract_hsv_histogram(new_image)]

plt.figure(figsize=(8, 4))
plt.bar(range(len(new_features[0])), new_features[0].flatten(), color='blue')
plt.axvline(x=255, color='red', linestyle='--', linewidth=2)
plt.xlabel("Saturation/Value Bin")
plt.ylabel("Frequency")
plt.title("Time Series Histogram")

new_features = scaler.transform(new_features)
prediction = knnPredict.predict(new_features)
print(f"Predicted Art Style: {prediction[0]}")

# Plot an example Hue Histogram
plt.show()
