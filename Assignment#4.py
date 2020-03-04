import pandas as pd
import sklearn.model_selection as ms
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

# Read the data
data = pd.read_csv("C:/Users/lenovo/Downloads/Iris.csv")
print data.head(10)

# Transform the dependent variable
enn = LabelEncoder()
data['Species_n'] = enn.fit_transform(data['Species'])
print data

# drop the dependent variable
X = data.drop(['Species', 'Species_n'], axis='columns')
print X
y = data['Species_n']
print y

# Split the data
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=100)
print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

# model with gini Index.
model_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
model_gini.fit(X_train, y_train)
print model_gini

# model with entropy Index.
model_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
model_entropy.fit(X_train, y_train)
print model_entropy

# Prediction
pred_gini = model_gini.predict(X_test)
print pred_gini
pred_entropy = model_entropy.predict(X_test)
print pred_entropy

# Accuracy
print ('Accuracy using Gini index', metrics.accuracy_score(y_test, pred_gini))
print ('Accuracy using Entropy index', metrics.accuracy_score(y_test, pred_entropy))















