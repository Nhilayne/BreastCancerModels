import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.nadam import Nadam
from keras.optimizer_v2.adam import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


PATH = "/Users/Iangs/Desktop/CST Term 4 Fall 2021/Comp 4948 Predictive Machine Learning/datasets/"
CSV = "haberman.csv"
dataset = pd.read_csv(PATH + CSV, index_col=None)

# show all columns
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(dataset.describe())
print(dataset.head())

#######
# preprocess data
#######

transformer = Binarizer(threshold=1, copy=False).fit(dataset[['survival']])
awards_binary = pd.DataFrame(data=transformer.transform(dataset[['survival']]), columns=["survival_binary"])
new_dataset = pd.concat(([dataset, awards_binary]), axis=1)

transformer = Binarizer(threshold=3, copy=False).fit(dataset[['nodes']])
awards_binary = pd.DataFrame(data=transformer.transform(dataset[['nodes']]), columns=["nodes_over_3"])
new_dataset = pd.concat(([new_dataset, awards_binary]), axis=1)

transformer = Binarizer(threshold=50, copy=False).fit(dataset[['age']])
awards_binary = pd.DataFrame(data=transformer.transform(dataset[['age']]), columns=["age_over_40"])
new_dataset = pd.concat(([new_dataset, awards_binary]), axis=1)


X = new_dataset[['age', 'year', 'nodes_over_3', 'age_over_40']]
y = new_dataset[['survival_binary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#######
# baseline logistic model
#######
logisticModel = None
average_accuracy = 0
num = 1
for i in range(0,num):
    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver='lbfgs')
    logisticModel.fit(X_train,y_train)
    y_pred = logisticModel.predict(X_test)

    # Show model coefficients and intercept.
    print("\nModel Coefficients: ")
    print("\nIntercept: ")
    print(logisticModel.intercept_)

    print(logisticModel.coef_)

    # Show confusion matrix and accuracy scores.
    confusion_matrix = pd.crosstab(np.array(y_test['survival_binary']), y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])

    average_accuracy += metrics.accuracy_score(y_test, y_pred)
    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix)

print('\nAverage Accuracy: ', average_accuracy/num)

#######
# baseline ann model
#######
neuralmodel = Sequential()
neuralmodel.add(Dense(25, input_dim=X.shape[1], kernel_initializer='glorot_uniform',activation='softplus'))
neuralmodel.add(Dense(15, kernel_initializer='glorot_uniform', activation='softplus'))
neuralmodel.add(Dense(5, kernel_initializer='glorot_uniform', activation='softplus'))
neuralmodel.add(Dense(1, activation='sigmoid'))
opt = Adam(learning_rate=0.001)
neuralmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)
mc = ModelCheckpoint('../best_model.h5', monitor='val_loss', mode='auto', verbose=0,
                     save_best_only=True)


# fit model
history = neuralmodel.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=25,
                    epochs=4000,
                    verbose=0,
                    callbacks=[es, mc])

# load the saved model
neuralmodel = load_model('../best_model.h5')


# Evaluate the model.
train_loss, train_acc = neuralmodel.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = neuralmodel.evaluate(X_test, y_test, verbose=0)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))
print('Train loss: %.3f, Test loss: %.3f' % (train_loss, test_loss))
# print('Train recall: %.3f, Test recall: %.3f' % (train_rec, test_rec))
# print('Train precision: %.3f, Test precision: %.3f' % (train_pre, test_pre))

# Plot loss learning curves.
plt.subplot(211)
plt.title('Cross-Entropy Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# Plot accuracy learning curves.
plt.subplot(212)
plt.title('Accuracy', pad=-40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


#######
# bagged logistic model
#######

lr = LogisticRegression(fit_intercept=True, solver='lbfgs')


def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)


def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    accuracy    = metrics.accuracy_score(y_test, predictions)
    recall      = metrics.recall_score(y_test, predictions, average='weighted')
    precision   = metrics.precision_score(y_test, predictions, average='weighted')
    f1          = metrics.f1_score(y_test, predictions, average='weighted')

    print("Accuracy:  " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall:    " + str(recall))
    print("F1:        " + str(f1))


# max_features means the maximum number of features to draw from X.
# max_samples sets the percentage of available data used for fitting.
bagging_clf = BaggingClassifier(lr, max_samples=0.2, max_features=X.shape[1], n_estimators=10)
baggedModel = bagging_clf.fit(X_train, y_train)
evaluateModel(baggedModel, X_test, y_test, "Bagged Model")


#######
# stacked model
#######


def getUnfitModels():
    models = list()
    models.append(LogisticRegression())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=4))
    return models

def evaluateModel(y_test, predictions, model):
    precision = round(precision_score(y_test, predictions),2)
    recall    = round(recall_score(y_test, predictions), 2)
    f1        = round(f1_score(y_test, predictions), 2)
    accuracy  = round(accuracy_score(y_test, predictions), 2)

    print("Precision:" + str(precision) + " Recall:" + str(recall) +\
          " F1:" + str(f1) + " Accuracy:" + str(accuracy) +\
          "   " + model.__class__.__name__)

def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfPredictions[colName] = predictions
    return dfPredictions, models

def fitStackedModel(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
stackedModel          = fitStackedModel(dfPredictions, y_test)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
dfValidationPredictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_test)
    colName = str(i)
    dfValidationPredictions[colName] = predictions
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
stackedPredictions = stackedModel.predict(dfValidationPredictions)
print("\n** Evaluate Stacked Model **")
evaluateModel(y_test, stackedPredictions, stackedModel)

