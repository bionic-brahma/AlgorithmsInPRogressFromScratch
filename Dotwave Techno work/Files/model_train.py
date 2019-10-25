import pickle
from sklearn import cross_validation
x=pickle.load(open("output_features.p","rb"))
y=pickle.load(open("output_labels.p","rb"))

from sklearn import svm


x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.25, random_state=0)


#clf = svm.SVC(kernel='rbf', C=10000)
clf = svm.SVC(kernel='rbf', C=1000).fit(x_train, y_train)
#clf.fit(x_train, y_train)
print(clf)

print('accuracy  =  '+str(100*clf.score(x_test, y_test)))
pickle.dump( clf, open( "class.p", "wb" ) )
