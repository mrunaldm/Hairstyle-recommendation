import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas
import numpy

label = [1,2,3,4,5]
shapeMapper = {"Heart":1,"Oblong":2,"Oval":3,"Round":4,"Square":5}

existing_data = pandas.read_csv("output.csv",header=0)
y = numpy.array(existing_data["Shape_Mapper"])
x = numpy.array(existing_data.ix[:,0:25])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=37)
clas = MLPClassifier(hidden_layer_sizes=100,solver="lbfgs",alpha=0.1,random_state=1)
cls = GaussianNB()
heartShape = 0
onlongShape = 0
ovalShape = 0
roundShape = 0
squareShape = 0
cls.fit(X_train,y_train)
resp = cls.predict(X_test)
print len(X_test)
for x in y_test:
    if x==1:
        heartShape = heartShape + 1
    elif x==2:
        onlongShape += 1
    elif x==3:
        ovalShape += 1
    elif x==4:
        roundShape += 1
    elif x==5:
        squareShape += 1
    else:
        print "Shape not defined"
print heartShape
print onlongShape
print ovalShape
print roundShape
print squareShape
print "Accuracy for NBC : {0} %".format(accuracy_score(resp,y_test)*100)
print "Confusion matrix"
print confusion_matrix(resp,y_test,labels=label)
clas.fit(X_train,y_train)
rep = clas.predict(X_test)
print "Accuracy for ANN : {0} %".format(accuracy_score(rep,y_test)*100)
print "Confusion matrix"
print confusion_matrix(rep,y_test,labels=label)
clf = svm.SVC(kernel='rbf',C=0.01)
clf.fit(X_train,y_train)
result = clf.predict(X_test)
print "Accuracy for SVM : {0} %".format(accuracy_score(result,y_test)*100)
print "Confusion matrix"
print confusion_matrix(result,y_test,labels=label)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
res = classifier.predict(X_test)

print "Accuracy for KNN : {0} % ".format(accuracy_score(res,y_test)*100)
print "Confusion matrix"
print confusion_matrix(res,y_test,labels=label)



