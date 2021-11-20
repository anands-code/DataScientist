import nltk
import plotly
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import sys
import os
import re
from sqlalchemy import create_engine
import pickle
from scipy.stats import gmean
# importing sklearn modules
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)

    X = df['message']

    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    print(y.head(1))
    y=y.astype(int)
    category_names = y.columns
    
    return X, y, category_names    
    

def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

# Custom transformer to extract the starting verb in a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            # Finding Verbs
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    
    # Starting Verb Transformer
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    To check the model performance we take mean of the fbeta_score, computed on each label.
    This helps to deal with multi-class/multi-label imbalanced cases by excluding negligigle f1score.
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true -> List of labels
        y_prod -> List of predictions
        beta -> Beta value to be used to calculate fscore metric
    
    Output:
        f1score -> Calculation geometric mean of fscore
    """
    
    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        model -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = model.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()