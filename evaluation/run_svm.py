from models.SklearnClassifier import SklearnClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import utils
import time

X = utils.get_feature_matrix('../data/doc2vec.npy')
y = utils.get_feature_matrix('../data/publications.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SklearnClassifier(SVC(), X_train, y_train)

start_time = time.time()
svm_classifier.train()
print('Training time: ', time.time() - start_time)

svm_classifier.evaluate(X_test, y_test, top_k=3)

