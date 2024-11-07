def union_intersection(lst1, lst2):
    union = list(set(lst1) | set(lst2))
    intersection = list(set(lst1) & set(lst2))
    return union, intersection

# Test with numeric lists
nums1 = [1, 2, 3, 4, 5]
nums2 = [3, 4, 5, 6, 7, 8]
print("Original lists:")
print(nums1)
print(nums2)
result = union_intersection(nums1, nums2)
print("\nUnion of said two lists:")
print(result[0])
print("\nIntersection of said two lists:")
print(result[1])

# Test with color lists
colors1 = ["Red", "Green", "Blue"]
colors2 = ["Red", "White", "Pink", "Black"]
print("\nOriginal lists:")
print(colors1)
print(colors2)
result = union_intersection(colors1, colors2)
print("\nUnion of said two lists:")
print(result[0])
print("\nIntersection of said two lists:")
print(result[1])
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris
iris = load_iris()   # Load the dataset
X = iris.data
y = iris.target
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb_classifier = GaussianNB()    # Initialize and fit the model
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)    # Make predictions
accuracy = accuracy_score(y_test, y_pred)    # Calculate performance metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
# Print performance metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
#Decision Trees

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
colnames = ['Buying_price', 'maint_cost', 'doors', 'persons', 'lug_boot', 'safety', 'decision']
data = pd.read_csv('car_evaluation.csv', names=colnames, header=None)
plt.figure(figsize=(5, 5))    # Plot the distribution of the 'decision' column
sns.countplot(x='decision', data=data)
plt.title('Count plot for decision')
data.decision.replace('vgood', 'acc', inplace=True)    # Simplify the categories in 'decision'
data.decision.replace('good', 'acc', inplace=True)
new_data = data.apply(LabelEncoder().fit_transform)   # Encode categorical features
x = new_data.drop(['decision'], axis=1)     # Separate features and target
y = new_data['decision']
# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(criterion="entropy")    # Initialize and fit the Decision Tree model
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)    # Make predictions
cm = confusion_matrix(y_test, dt_pred)     # Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.show()

#Support Vector Machine

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = load_digits()     # Load the dataset
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=4)
classifier = SVC(kernel="linear")    # Initialize and train the SVM classifier
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)    # Make predictions
accuracy = accuracy_score(y_test, y_pred) * 100     # Calculate accuracy and confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Accuracy for SVM is:", accuracy)    # Print the results
print("Confusion Matrix:")
print(confusion_mat)
#K-Means Clustering  

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the dataset
X, y = load_iris(return_X_y=True)
# Initialize and fit KMeans
kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(X)
pred = kmeans.predict(X)
# Plot the results
plt.figure(figsize=(12, 5))
# Plot for the first two features
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=cm.Accent)
plt.grid(True)
# Plot cluster centers
for center in kmeans.cluster_centers_:
    center = center[:2]
    plt.scatter(center[0], center[1], marker='^', c='red')

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
# Plot for the last two features
plt.subplot(1, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=pred, cmap=cm.Accent)
plt.grid(True)

# Plot cluster centers
for center in kmeans.cluster_centers_:
    center = center[2:4]
    plt.scatter(center[0], center[1], marker='^', c='red')

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()
