import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Training data: Weight (kg), Height (cm)
X_train = np.array([[55, 160], [58, 162], [60, 165], [63, 170], [65, 172], [70, 175]])
y_train = np.array(["P", "P", "P", "L", "L", "L"])  # Labels: P = Perempuan, L = Laki-laki

# New data to classify
X_new = np.array([[61, 167]])

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(f"KNN prediction: {knn.predict(X_new)[0]}")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print(f"Naive Bayes prediction: {nb.predict(X_new)[0]}")

# Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print(f"SVM prediction: {svm.predict(X_new)[0]}")

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(f"Decision Tree prediction: {dt.predict(X_new)[0]}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f"Random Forest prediction: {rf.predict(X_new)[0]}")
