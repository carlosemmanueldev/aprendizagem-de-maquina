import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

labels = ['class', 'X1', 'X2' , 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']

data = pd.read_csv('accent-mfcc-data-1.csv', header=0, names=labels)

X_train, X_test, y_train, y_test = train_test_split(data[labels[1:]], data['class'], test_size=0.5, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=7)

neigh.fit(X_train, y_train)

neigh.predict(X_test)

print("Taxa de acerto: ", neigh.score(X_test, y_test))