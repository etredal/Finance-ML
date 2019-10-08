import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

api_key = "W5BMIZIWT1R06OIS"

#My needed stock
symbolText = "MSFT"
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbolText, outputsize = 'full')

data.to_excel(symbolText + "-output.xlsx")

close_data = data['4. close'].to_list() #Now a list of floats
date_data = pd.to_datetime(data.index).to_list() #Now a list of Timestamps
date_data_final = list(date_data)

#Convert Timestamp into int based on how far it is from the first day
first_day = date_data[0]
for i in range(0,len(date_data)):
    date_data[i] = (date_data[i] - first_day).days
    date_data_final[i] = (date_data_final[i] - first_day).days

#Adds new stock values for the missing days
total_increase = 0
for i in range(0,len(date_data) - 1):
    discrete_increase = 1
    while (date_data[i] + discrete_increase != date_data[i + 1]):
        total_increase += 1
        date_data_final.insert(i + total_increase,i + total_increase) #New day numbers
        
        close_data.insert(i + total_increase, round((close_data[i + total_increase - discrete_increase] + close_data[i + total_increase])/2, 2)) #Average of the days stocks
        
        discrete_increase += 1



#Machine Learning train data
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Accuracy of Logitistic Regression: " + str(acc_log))

#Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("Accuracy of Support Vector Machine: " + str(acc_svc))

#K-Nearest Neighbors Algorithm
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("Accuracy of kNN: " + str(acc_knn))

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("Accuracy of Guassian Naive Bayes: " + str(acc_gaussian))

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print("Accuracy of Perceptron: " + str(acc_perceptron))

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("Linear SVC: " + str(acc_linear_svc))

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("SGD: " + str(acc_sgd))

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Decision Tree: " + str(acc_decision_tree))

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Random Forest: " + str(acc_random_forest))