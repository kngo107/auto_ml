import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from auto_ml.utils import get_boston_dataset
from auto_ml.utils import get_titanic_binary_classification_dataset

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

from sklearn import model_selection

def plot_metric(metric, data):

    #Set title and Label:
    plt.title("{} Score of per dataset".format(metric.upper()))
    plt.xlabel('Dataset')
    plt.ylabel(metric)

    bar_width = 0.35
    opacity = 0.8
    n_groups = 0
    auto_ml_score = []
    mlp_score = []
    x_label = []
    #==================================================================
    for dataset_name,scores in data.iteritems():
        n_groups +=1
        x_label.append(dataset_name)
        auto_ml_score.append(scores['auto_ml'])
        mlp_score.append(scores['mlp'])
    index = np.arange(n_groups)
    rects1 = plt.bar(index, tuple(auto_ml_score), bar_width,
                    alpha=opacity,
                    color='k',
                    label='auto_ml')
    rects2 = plt.bar(index + bar_width, tuple(mlp_score), bar_width,
                    alpha=opacity,
                    color='r',
                    label='mlp')
    plt.xticks(index + bar_width, tuple(x_label))
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_data_for_testing(dataset_name):
    np.random.seed(0)
    #Get metrics for all datasets
    if 'boston' in dataset_name:
        #SPECIFIC FIELDS============================================
        #regressor problem
        #506 instasnces, 13 attributes (numeric/categorical)
        problem_type = 'regressor'
        output = 'MEDV'
        column_descriptions = {
                output: 'output',
                'CHAS': 'categorical'
                }
        raw_data = load_boston()
        #===========================================================
        df = pd.DataFrame(raw_data.data)
        df.columns = raw_data.feature_names
        df[output] = raw_data['target']
        train_set, test_set = model_selection.train_test_split(df, test_size=0.4, random_state=42)

    elif dataset_name == 'diabetes':
        #SPECIFIC FIELDS============================================
        #regressor problem
        #442 instasnces, 10 attributes (numeric)
        problem_type = 'regressor'
        output = 'DIA'
        column_descriptions = {
                output: 'output'
                }
        raw_data = load_diabetes()
        #===========================================================
        df = pd.DataFrame(raw_data.data)
        df.columns = raw_data.feature_names
        df[output] = raw_data['target']
        train_set, test_set = model_selection.train_test_split(df, test_size=0.4, random_state=42)

    elif dataset_name == 'linnerud':
        #SPECIFIC FIELDS============================================
        #regressor problem
        #20 instasnces, 3 attributes (numeric)
        problem_type = 'regressor'
        output = 'LIN'
        column_descriptions = {
                output: 'output'
                }
        raw_data = load_linnerud()
        #===========================================================
        df = pd.DataFrame(raw_data.data)
        df.columns = raw_data.feature_names
        df[output] = raw_data['target'][:,0]
        train_set, test_set = model_selection.train_test_split(df, test_size=0.4, random_state=42)

    elif dataset_name == "titanic":
        problem_type = "classifier"
        output = 'survived'
        column_descriptions = {
            output: 'output'
            , 'sex': 'categorical'
            , 'embarked': 'categorical'
            , 'pclass': 'categorical'
        }
        train_set, test_set = get_titanic_binary_classification_dataset()
    elif dataset_name == "songs":
        problem_type = "regressor"
        output = 'output'
        column_descriptions = {
                output: 'output'
                }

        columns = ['output']
        for i in range(89):
            columns.append("input{}".format(i))

        dir_name = os.path.abspath(os.path.dirname(__file__))
        file_name = os.path.join(dir_name, "data/YearPredictionMSD.txt")

        df_songs = pd.read_csv(file_name, names = columns)

        train_set = df_songs.iloc[:463715]
        test_set  = df_songs.iloc[463715:]

    else:
        print "{} doesn't existed in the database".format(dataset_name)
        exit()

    return problem_type, output, column_descriptions, train_set, test_set
