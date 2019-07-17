# general import
from time import time
# scikit learn imports

# vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections

# classifiers
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

# gridsearch
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


param_vect_countvect= {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2), (1,3), (2,2), (2,3)),
    'features__pipe__vect__min_df': (1,2,3),
}

param_vect_tfidf= {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2),(1,3),(2,2), (2,3)),
    'features__pipe__vect__min_df': (1,2,3),
    #Tune the type of norms to be used 
    #'features__pipe__vect__norm':('l1','l2', None)
}


# multinomial naive bayes parameters
param_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
}

param_xgb = {
    'clf_max_depth':(4,5,6,7,8),
    'clf_refresh_leaf':(0,1),
    'clf_colsample_bytree': (0.3, 0.4, 0.5, 0.6),
    'clf_seed':(1,2,3),
    'clf_subsample':(0.3,0.5,0.7)

}

# logistic regression parameters
param_logreg = {
    'clf__C': (0.25, 0.5, 1.0),
    'clf__penalty': ('l1', 'l2')
}


# linear SVC
param_linearsvc = {'clf__C': (0.05, 0.10, 0.15, 0.20,0.25)}

# Random Forest 
param_RandF = {
    'clf__max_depth': (2,3,4),
    'clf__n_estimators': (200,300),
    'clf__random_state':(0,1,2,3)
    
}


# partial code from: https://github.com/bertcarremans/TwitterUSAirlineSentiment/blob/master/source/Predicting%20sentiment%20with%20text%20features.ipynb
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols
    def transform(self, X, **transform_params):
        return X[self.cols]
    def fit(self, X, y=None, **fit_params):
        return self
    
def grid_vect(clf, parameters_clf, text_train, class_train, text_test, class_test, parameters_text=None, vect=None):
    features = FeatureUnion([('pipe', Pipeline([('processedtext', ColumnExtractor(cols='processed_string')), ('vect', vect)]))], n_jobs=-1)
    pipeline = Pipeline([
        ('features', features)
        , ('clf', clf)])
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    t0 = time()
    grid_search.fit(text_train, class_train)
    print("done in %0.3fs" % (time() - t0))
    print("\n")
    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name])) 
    print("Cross Validated Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(text_test, class_test))
    print("\n")
    print("Classification Report Test Data")
    print("\n")
    print(classification_report(class_test, grid_search.best_estimator_.predict(text_test)))
    print("Accuracy Score: ", accuracy_score(class_test, grid_search.best_estimator_.predict(text_test)))
                        
    return grid_search

def feature_pipe(Vectorizer, classifier, text_train, class_train, data):
    features = FeatureUnion([('pipe', Pipeline([('processedtext', ColumnExtractor(cols='processed_string')), ('vect', Vectorizer)]))], n_jobs=-1)
    pipeline = Pipeline([('features', features), ('clf', classifier)])   
    best_model = pipeline.fit(text_train, class_train)
    return (best_model.predict(data).tolist())        
