# Stacking Classifiers
from sklearn import model_selection
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
import warnings
# suppress warnings in sklearn 
warnings.filterwarnings("ignore", category=FutureWarning)

# Voting Classifiers
from sklearn.ensemble import VotingClassifier

def scoring(classifiers, classifier_names, X_train, y_train):
    print('5-fold cross validation:\n')
    for classifiers, clf_name in zip(classifiers , classifier_names):
        scores = model_selection.cross_val_score(classifiers, X_train, y_train, cv=5, scoring='accuracy')
        print("5-fold cross validated Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), clf_name))
