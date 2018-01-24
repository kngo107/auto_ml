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
    opacity = 1
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
    plt.xticks((index + bar_width - (bar_width/2)), tuple(x_label))
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_dataset_num_metric(metric, data, total_training_size):

    #Set title and Label:
    plt.title("{} vs Training Space".format(metric.upper()))
    plt.xlabel('Training Space Percentage. Total training size is {}'.format(total_training_size))
    plt.ylabel(metric)

    bar_width = 0.2
    opacity = 1
    n_groups = 0
    auto_ml_score = []
    mlp_score = []
    x_label = []
    
    #==================================================================
    for dataset_name,scores in sorted(data.iteritems()):
        n_groups +=1
        x_label.append(dataset_name)
        auto_ml_score.append(scores['auto_ml'])
        mlp_score.append(scores['mlp'])
    index = np.arange(n_groups)
    plt.plot(x_label,auto_ml_score, marker="v")
    plt.plot(x_label,mlp_score, marker="v")
    plt.legend(['auto_ml','mlp'], loc='upper left')
    plt.tight_layout()
    plt.show()

    
def get_dataset_size_list(dataset):
    output_list = []
    if 'boston' in dataset:
        for i in range(10,100,10):
            output_list.append("{} {}".format(dataset,i))
    if 'songs' in dataset:
        for i in range(10,50,5):
            output_list.append("{} {}".format(dataset,i))
    else:
        print "Your dataset is not existed in the database"
        exit()
    return output_list

def get_data_for_testing(dataset_name):
    np.random.seed(0)
    training_fraction = 100
    if _get_num_from_string(dataset_name) is not None:
        training_fraction = _get_num_from_string(dataset_name) 
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
        total_training_size = train_set.shape[0]
        train_set = train_set.iloc[:int((training_fraction/100.0)*train_set.shape[0])]

    elif 'diabetes' in dataset_name:
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
        total_training_size = train_set.shape[0]
        train_set = train_set.iloc[:int((training_fraction/100.0)*train_set.shape[0])]

    elif 'linnerud' in dataset_name:
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
        total_training_size = train_set.shape[0]
        train_set = train_set.iloc[:int((training_fraction/100.0)*train_set.shape[0])]

    elif "titanic" in dataset_name:
        problem_type = "classifier"
        output = 'survived'
        column_descriptions = {
            output: 'output'
            , 'sex': 'categorical'
            , 'embarked': 'categorical'
            , 'pclass': 'categorical'
        }
        train_set, test_set = get_titanic_binary_classification_dataset()
        total_training_size = train_set.shape[0]
        train_set = train_set.iloc[:int((training_fraction/100.0)*train_set.shape[0])]
    elif "songs" in dataset_name:
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
        total_training_size = train_set.shape[0]
        train_set = train_set.iloc[:int((training_fraction/100.0)*train_set.shape[0])]

    else:
        print "{} doesn't existed in the database 1".format(dataset_name)
        exit()

    return problem_type, output, column_descriptions, train_set, test_set, total_training_size

def _get_num_from_string(dataset):
    numbers = [int(s) for s in dataset.split() if s.isdigit()]
    if numbers:
        return numbers[0]
    return None
