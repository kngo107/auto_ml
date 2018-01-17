from auto_ml import Predictor
from auto_ml.utils_performance import plot_metric
from auto_ml.utils_performance import get_data_for_testing

import copy

import pandas as pd
import numpy as np
import json
import os
import datetime

from sklearn import metrics
from sklearn import neural_network
import matplotlib.pyplot as plt


def _saveObj(obj, name):
    if not os.path.exists(os.path.dirname('./results/' + name)):
        os.makedirs(os.path.dirname('./results/' + name))
    with open('./results/' + name + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +'.txt', 'wb+') as f:
        json.dump(obj, f)
        #for a performance boost use 
        #pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def _testWithData(dataset):
    count = 1
    metrics = _getMetrics(dataset, 1)
    _saveObj(metrics, "basic_test_")
    for metric, data in metrics.iteritems():
        plt.figure(count)
        count +=1
        plot_metric(metric,data)

def _getMetrics(dataset, n_fold):
    results = dict()
    #Get metrics for all datasets
    while (dataset):
        dataset_name = dataset[0]

        problem_type, output, column_descriptions, train_set, test_set = get_data_for_testing(dataset_name)


        if problem_type == 'regressor':
            result = {'mse':{dataset_name:{'auto_ml':0,'mlp':0}},
                      'r2':{dataset_name:{'auto_ml':0,'mlp':0}}}
        else:
            result = {'accuracy':{dataset_name:{'auto_ml':0,'mlp':0}},
                      'f1':{dataset_name:{'auto_ml':0,'mlp':0}}}
                        
        
        average_factor = 1.0/n_fold
        for i in range(n_fold):

            ml_predictor = Predictor(type_of_estimator=problem_type, column_descriptions=column_descriptions)
           # train_set_transformed = ml_predictor.transform_only(train_set)
            
            ml_predictor.train(train_set, verbose=False, compare_all_models = True)

            y_auto_ml_predicted = ml_predictor.predict(test_set)

            train_set_x = ml_predictor.transform_only(train_set.drop(output,axis=1))


            if problem_type == 'regressor':
                train_set_y = np.asarray(train_set[output], dtype=np.int64)
                mlp = neural_network.MLPRegressor(hidden_layer_sizes=(20,20,20), max_iter = 500)
                mlp.fit(train_set_x,train_set_y)
            else:
                train_set_y = np.asarray(train_set[output], dtype=np.int64)
                mlp = neural_network.MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter = 500)
                mlp.fit(train_set_x,train_set_y)

            test_set_x = test_set.drop(output,axis=1)
            y_mlp_predicted = mlp.predict(ml_predictor.transform_only(test_set_x))

            if problem_type == 'regressor':
                result['mse'][dataset_name]['auto_ml'] += (metrics.mean_squared_error(test_set[output],y_auto_ml_predicted)*average_factor)
                result['r2'][dataset_name]['auto_ml'] += (metrics.r2_score(test_set[output],y_auto_ml_predicted)*average_factor)
                result['mse'][dataset_name]['mlp'] += (metrics.mean_squared_error(test_set[output], y_mlp_predicted)*average_factor)
                result['r2'][dataset_name]['mlp'] += (metrics.r2_score(test_set[output],y_mlp_predicted)*average_factor)
            else:
                result['accuracy'][dataset_name]['auto_ml'] += (metrics.accuracy_score(test_set[output],y_auto_ml_predicted)*average_factor)
                result['f1'][dataset_name]['auto_ml'] += (metrics.f1_score(test_set[output],y_auto_ml_predicted, average='macro')*average_factor)
                result['accuracy'][dataset_name]['mlp'] += (metrics.accuracy_score(test_set[output], y_mlp_predicted)*average_factor)
                result['f1'][dataset_name]['mlp'] += (metrics.f1_score(test_set[output],y_mlp_predicted, average='macro')*average_factor)


        #Take average
        for test, data in result.iteritems():
            if not test in results:
                results[test] = dict()
            results[test][dataset_name] = data[dataset_name]  
                

        del dataset[0]
    return results

_testWithData(['boston'])
