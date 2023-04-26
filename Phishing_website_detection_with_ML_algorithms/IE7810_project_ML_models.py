import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import keras
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
from keras.models import Model
from sklearn import metrics
from sklearn.svm import SVC


#loading dataset
#we collected this data based ob feature extraction code
dfurl = pd.read_csv('urldata.csv')

# visualizing dataset
# data distribution
dfurl.hist(bins = 50,figsize = (15,15))

#Correlation heatmap

plt.figure(figsize=(15,13))
sns.heatmap(dfurl.corr())
plt.show()

#Dropping the Domain column, which we do not need that during modeling
df = dfurl.drop(['Domain'], axis = 1).copy()

# shuffling the dataset for test and train split
df = df.sample(frac=1).reset_index(drop=True)
y = df['Label']
X = df.drop('Label',axis=1)

# Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
# Creating empty lists to store the model performance results after running each model.
ML_Model = []
acc_train = []
acc_test = []

#function to call for storing the results
def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))

# Decision Tree model
tree = DecisionTreeClassifier(max_depth = 5)
tree.fit(X_train, y_train)
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)
accuracy_train_tree = accuracy_score(y_train,y_train_tree)
accuracy_test_tree = accuracy_score(y_test,y_test_tree)
print("Decision Tree: Training Data Accuracy: {:.3f}".format(accuracy_train_tree))
print("Decision Tree: Test Data Accuracy: {:.3f}".format(accuracy_test_tree))

#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
#Result storing
storeResults('Decision Tree', accuracy_train_tree, accuracy_test_tree)

# Random Forest model
# instantiate the model
forest = RandomForestClassifier(max_depth=5)
# fit the model
forest.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)
#computing the accuracy of the model performance
accuracy_train_forest = accuracy_score(y_train,y_train_forest)
accuracy_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Training Data Accuracy: {:.3f}".format(accuracy_train_forest))
print("Random forest: Test Data Accuracy: {:.3f}".format(accuracy_test_forest))
#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

storeResults('Random Forest', accuracy_train_forest, accuracy_test_forest)

# Multilayer Perceptrons model
# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))
mlp.fit(X_train, y_train)
# predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)
# computing the accuracy of the model performance
accuracy_train_mlp = accuracy_score(y_train,y_train_mlp)
accuracy_test_mlp = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons: Training Data Accuracy: {:.3f}".format(accuracy_train_mlp))
print("Multilayer Perceptrons: Test Data Accuracy: {:.3f}".format(accuracy_test_mlp))

storeResults('Multilayer Perceptrons', accuracy_train_mlp, accuracy_test_mlp)

#XGBoost Classification model
# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)
#computing the accuracy of the model performance
accuracy_train_xgb = accuracy_score(y_train,y_train_xgb)
accuracy_test_xgb = accuracy_score(y_test,y_test_xgb)

print("XGBoost: Training Data Accuracy: {:.3f}".format(accuracy_train_xgb))
print("XGBoost : Test Data Accuracy: {:.3f}".format(accuracy_test_xgb))

storeResults('XGBoost', accuracy_train_xgb, accuracy_test_xgb)

#building autoencoder model

input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)

encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim-2), activation='relu')(code)

decoder = Dense(int(encoding_dim), activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

#compiling the model
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#Training the model
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2)
accuracy_train_auto = autoencoder.evaluate(X_train, X_train)[1]
accuracy_test_auto = autoencoder.evaluate(X_test, X_test)[1]

print('\nAutoencoder: Training Data Accuracy: {:.3f}' .format(accuracy_train_auto))
print('Autoencoder: Test Data Accuracy: {:.3f}' .format(accuracy_test_auto))

storeResults('AutoEncoder', accuracy_train_auto, accuracy_test_auto)

#Support vector machine model
svm = SVC(kernel='linear', C=1.0, random_state=12)
#fit the model
svm.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)
#computing the accuracy of the model performance
accuracy_train_svm = accuracy_score(y_train,y_train_svm)
accuracy_test_svm = accuracy_score(y_test,y_test_svm)

print("SVM: Training Data Accuracy: {:.3f}".format(accuracy_train_svm))
print("SVM : Test Data Accuracy: {:.3f}".format(accuracy_test_svm))

storeResults('SVM', accuracy_train_svm, accuracy_test_svm)

#creating dataframe for model comparison
results = pd.DataFrame({ 'ML Model': ML_Model,
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results
# sorting performance results high to low
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
