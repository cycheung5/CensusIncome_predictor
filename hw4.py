import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from scipy.spatial import distance
import random 



def getGamma(X_data):
    # Need to get the gamma0 value
    df = pd.DataFrame(X_data)
    x = df.sample(frac=0.5, random_state=1)
    X = np.array(x)
    gamma_arr = []
    for i in range(0, len(X) - 1):
        for j in range(i + 1, len(X)):
            result = distance.sqeuclidean(X[i], X[j])
            gamma_arr.append(result)
    mean_val = np.mean(gamma_arr)
    gamma = 1 / mean_val
    return gamma




def kfoldsplit(cvalue, X_data, y):
    accuracy_val = []
    clf = SVC(C=cvalue, kernel='linear')
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X_data):
        X_train = X_data[train_index]
        X_test = X_data[test_index]
        y_train = y[train_index]
        y_test = y[test_index] 
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        # Accuracy
        score = metrics.accuracy_score(y_test, y_predict)
        accuracy_val.append(score)
    nparray = np.asarray(accuracy_val)
    return np.mean(nparray)



def rbf_kfold(X_data, y):
    # Need to get the gamma0 value
    accuracy_val = []
    #report = np.zeros((5,5))
    c_list = [0.01, 0.1, 1, 10, 100]
    gamma0 = getGamma(X_data)
    gamma1 = gamma0 * 0.01
    gamma2 = gamma0 * 0.1
    gamma4 = gamma0 * 10
    gamma5 = gamma0 * 100
    gamma_list = [gamma1, gamma2, gamma0, gamma4, gamma5]
    clf = SVC(C=c_list[1], kernel='rbf', gamma=gamma_list[3])
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X_data):
        X_train = X_data[train_index]
        X_test = X_data[test_index]
        y_train = y[train_index]
        y_test = y[test_index] 
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        # Accuracy
        score = metrics.accuracy_score(y_test, y_predict)
        accuracy_val.append(score)
    nparray = np.asarray(accuracy_val)
    return np.mean(nparray)

def XGboost_kfold(X_data, y, eta):
    accuracy_val = []
    model = XGBClassifier(min_child_weight=eta)
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X_data):
        X_train = X_data[train_index]
        X_test = X_data[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        # Accuracy
        score = metrics.accuracy_score(y_test, y_predict)
        accuracy_val.append(score)
    nparray = np.asarray(accuracy_val)
    return np.mean(nparray)


def main():

    # Load the training data and test data
    training_data = np.genfromtxt("adult.data", delimiter=",", dtype=np.object, autostrip=True)
    testing_data = np.genfromtxt("adult.test", delimiter=",", dtype=np.object, autostrip=True)

    # Fill in the mising data
    imp = SimpleImputer(missing_values=b'?', strategy='most_frequent')
    train_data = imp.fit_transform(training_data)
    test_data = imp.transform(testing_data)

    # Encode categorical data

    train_df = pd.DataFrame(train_data, columns=["Age", "WorkClass", "Fnlwgt", "Education", "Education-num", "Martial-status", "Occupation", "Relationship", "Race", "Sex", "Capital-gain", "Capital-loss", "hours-per-week", "Native-country", "Salary"])
    test_df = pd.DataFrame(test_data, columns = ["Age", "WorkClass", "Fnlwgt", "Education", "Education-num", "Martial-status", "Occupation", "Relationship", "Race", "Sex", "Capital-gain", "Capital-loss", "hours-per-week", "Native-country", "Salary"])
    enc = OrdinalEncoder()
    train_df[["WorkClass","Education", "Martial-status", "Occupation", "Relationship", "Race", "Sex", "Native-country", "Salary"]] = enc.fit_transform(train_df[["WorkClass","Education", "Martial-status", "Occupation", "Relationship", "Race", "Sex", "Native-country", "Salary"]])
    test_df[["WorkClass","Education", "Martial-status", "Occupation", "Relationship", "Race", "Sex", "Native-country", "Salary"]] = enc.transform(test_df[["WorkClass","Education", "Martial-status", "Occupation", "Relationship", "Race", "Sex", "Native-country", "Salary"]])
    
    train = train_df.to_numpy()
    test = test_df.to_numpy()
    
    # Convert them to floats
    train = train.astype(np.float)
    test = test.astype(np.float)

    

    # Split up the train_data and test_data 
    x_train = train[:, 0:14]
    y_train = train[:, 14]
    x_test = test[:, 0:14]
    y_test = test[:, 14]


    # Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #SVM
    #1.1
    #Create a svm classifier
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    y_predict = clf.predict(x_train)
    print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
    print("Confusion matrix:", metrics.confusion_matrix(y_test, y_hat))

    clf = SVC(kernel='rbf')
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_train)
    print("Accuracy:",metrics.accuracy_score(y_train, y_hat))
    print("Confusion matrix:", metrics.confusion_matrix(y_train, y_hat))

    #1.2
    #avg_score = kfoldsplit(100, x_train, y_train)
    #print(avg_score)

    clf = SVC(C=0.1, kernel='linear')
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
    print("Confusion matrix:", metrics.confusion_matrix(y_test, y_hat))


    #1.3

    #val = rbf_kfold(x_train, y_train)


    clf = SVC(C=10, kernel='rbf', gamma=0.035399836257649225)
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
    print("Confusion matrix:", metrics.confusion_matrix(y_test, y_hat))


    #2.1
    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    #y_predict = model.predict(x_train)
    print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
    print("Confusion matrix:", metrics.confusion_matrix(y_test, y_predict))


    #val = XGboost_kfold(x_train, y_train, 50)
    #print(val)
    model = XGBClassifier(max_depth=4, learning_rate=0.1, gamma=0.25, min_child_weight=1, colsample_bytree=0.25, reg_alpha=0, reg_lambda=0.01)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
    print("Confusion matrix:", metrics.confusion_matrix(y_test, y_predict))

   

if __name__ == "__main__":
    main()