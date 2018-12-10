import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import argparse
import multiprocessing

import librosa
import librosa.display
import io 
import csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import pickle
from sklearn.externals import joblib

cores = multiprocessing.cpu_count() - 2

    
#def get_best_rf_params(random_state, param_grid, cores, X_train, y_train):
#    model_rf = RandomForestRegressor(random_state = random_state)
#
#    model_rf_cv = GridSearchCV(estimator = model_rf, param_grid = param_grid, 
#                              cv = 10, n_jobs = cores, verbose = 2)
#    
#    model_rf_cv = model_rf_cv.fit(X_train, y_train.ravel())
#    
#    return model_rf_cv.best_params_
#
#    
#def train_final_model(random_state, param_grid, cores, X_train, y_train):
#    best_parameters = get_best_rf_params(random_state, param_grid, cores, X_train, y_train)
#    
#    return best_parameters, RandomForestRegressor(random_state = random_state, 
#                                                  bootstrap = best_parameters["bootstrap"],
#                                                  criterion = best_parameters["criterion"],
#                                                  max_depth = best_parameters["max_depth"],
#                                                  min_samples_leaf = best_parameters["min_samples_leaf"],
#                                                  min_samples_split = best_parameters["min_samples_split"],
#                                                  n_estimators = best_parameters["n_estimators"], 
#                                                  verbose = 2,
#                                                  n_jobs = cores).fit(X_train, y_train.ravel())
    
    
def simple_rf(random_state, bootstrap, n_estimators, max_depth, max_features, cores, X_train, y_train):
    return RandomForestRegressor(random_state = random_state, 
                                 bootstrap = bootstrap,
                                 n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 verbose = 2,
                                 n_jobs = cores).fit(X_train, y_train.ravel())
    
#def get_prediction(random_state, param_grid, cores, X_train, y_train, X_test):
#    _, model = train_final_model(random_state, param_grid, cores, X_train, y_train)
#    
#    return model.predict(X_test)


def get_ccc(y_true, y_pred):
    return ccc(y_true, y_pred.reshape((y_pred.shape[0], 1)))


def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)
    
    rho,_ = pearsonr(y_pred,y_true)
    
    std_predictions = np.std(y_pred)
    
    std_gt = np.std(y_true)
    
    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)
    
    return ccc, rho
    


    
if __name__ == "__main__":
#    path_ = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Result/'
#    X_train = np.load(path_+'Train_spectogram.npy')
#    y_train = np.load(path_ + "Train_annotation.npy")
#    
#    X_valid = np.load(path_ + "Val_spectogram.npy")
#    y_valid = np.load(path_ + "Val_annotation.npy")
#    
#    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
#    
#    
#    param_grid = {
#        'bootstrap': [True, False], # [True, False]
#        'criterion': ["mse", "mae"], # ["mse", "mae"]
#        'max_depth': [5, 10, 20, 40], # [5, 10, 20, 40]
#        'max_features': ["auto", "sqrt", "log2"],
#        'min_samples_leaf': [3, 5, 10],
#        'min_samples_split': [2, 10, 40, 100], # [2, 10, 40, 100]
#        'n_estimators': [200, 500, 1000, 2000] # 200, 500, 1000, 2000
#    }
#    
#    params_of_model_rf, model_rf = train_final_model(2018, param_grid, cores, X_train, y_train)
#    
#    prediction_t = model_rf.predict(X_test)
#    # prediction = get_prediction(2018, param_grid, cores, X_train, y_train, X_test)
#    ccc_on_internal_test_set = get_ccc(y_test, prediction_t)
#    
#    prediction_val = model_rf.predict(X_valid)
#    ccc_on_internal_validation_set = get_ccc(y_valid, prediction_val)
#
#    # save the model
#    # save_model_rf = pickle.dumps(model_rf)
#    model_name = 'C:/Users/Raha/Research_data/OMG_empathy/omg_emp_predictor2.joblib.pkl'
#    _ = joblib.dump(model_rf, model_name, compress=9)
#    
#    
#    load_model_rf = joblib.load(model_name)
    
    
    # parameters 
    random_state, bootstrap, n_estimators, max_depth, max_features = 2018, True, 500, 30, "auto"
    
    
    
    # Landmarks (subject) - rf based empathy prediction 
    path_ = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/"
    s_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/"
    
    y_train_annotation = np.load(path_ + "Train_annotation.npy")
    y_val_annotation = np.load(path_ + "Val_annotation.npy")

    X_train_land_sub = pd.read_csv(path_ + "training_subs_landmarks.csv", header = None)
    X_val_land_sub = pd.read_csv(path_ + "validation_subs_landmarks.csv", header = None)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_land_sub, y_train_annotation, test_size=0.1)

    model_rf_land_sub = simple_rf(random_state, bootstrap, n_estimators, max_depth, max_features, cores, X_train, y_train)
    prediction_test_land_sub = model_rf_land_sub.predict(X_test)
    
    prediction_val_land_sub = model_rf_land_sub.predict(X_val_land_sub)
    ccc_val_land_sub = get_ccc(y_val_annotation, prediction_val_land_sub)
    
    # save model and prediction 
    model_name = s_path + 'omg_emp_model_rf_land_sub.joblib.pkl'
    _ = joblib.dump(model_rf_land_sub, model_name, compress=9)
    np.savetxt(s_path + 'prediction_val_land_sub.csv', prediction_val_land_sub, fmt='%.2f', delimiter=',')

    # omg_emp_model_rf_land_sub = joblib.load(model_name)
    
    
    # Landmarks (actor + subject) - rf based empathy prediction 
    X_train_land_act = pd.read_csv(path_ + "trianing_acts_landmarks.csv", header = None)    
    X_val_land_act = pd.read_csv(path_ + "validation_acts_landmarks.csv", header = None)
    
    # Actor + Subject
    X_train_act_sub = np.hstack([X_train_land_act, X_train_land_sub])
    X_val_act_sub = np.hstack([X_val_land_act, X_val_land_sub])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_act_sub, y_train_annotation, test_size=0.1)

    model_rf_land_act_sub = simple_rf(random_state, bootstrap, n_estimators, max_depth, max_features, cores, X_train, y_train)
    prediction_test_land_act_sub = model_rf_land_act_sub.predict(X_test)
    
    prediction_val_land_act_sub = model_rf_land_act_sub.predict(X_val_act_sub)
    ccc_val_land_act_sub = get_ccc(y_val_annotation, prediction_val_land_act_sub)
    
    # save model and prediction 
    model_name = s_path + 'omg_emp_model_rf_land_act_sub.joblib.pkl'
    _ = joblib.dump(model_rf_land_act_sub, model_name, compress=9)
    np.savetxt(s_path + 'prediction_val_land_act_sub.csv', prediction_val_land_act_sub, fmt='%.2f', delimiter=',')

    # omg_emp_model_rf_land_sub = joblib.load(model_name)
    
    
    
    
    # Deep Features
    X_train_deep_features = np.load(path_ + "deep_features_training.npy") # change this file to Deep feature file
    X_val_deep_features = np.load(path_ + "deep_features_validation.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_deep_features, y_train_annotation, test_size=0.1)

    model_rf_deep_features = simple_rf(random_state, bootstrap, n_estimators, max_depth, max_features, cores, X_train, y_train)
    prediction_test_deep_features = model_rf_deep_features.predict(X_test)
    
    prediction_val_deep_features = model_rf_deep_features.predict(X_val_land_sub)
    ccc_val_deep_features = get_ccc(y_val_annotation, prediction_val_deep_features)
    
    # save model and prediction 
    model_name = s_path + 'omg_emp_model_rf_deep_features.joblib.pkl'
    _ = joblib.dump(model_rf_deep_features, model_name, compress=9)
    np.savetxt(s_path + 'prediction_val_deep_features.csv', prediction_val_deep_features, fmt='%.2f', delimiter=',')

    omg_emp_model_rf_land_sub = joblib.load(model_name)
    
    
    
    # Spectorgram + Landmarkds + DeepFeatures 
    X_train_spec = np.load(path_+'Train_spectogram.npy')
    X_val_spec = np.load(path_ + "Val_spectogram.npy")
    
    # Spec + Actor + Subject + Deep Features 
    X_train_spec_act_sub_df = np.hstack([X_train_spec, X_train_land_act, X_train_land_sub, X_train_deep_features])
    X_val_spec_act_sub_df = np.hstack([X_val_spec, X_val_land_act, X_val_land_sub, X_val_deep_features])

    X_train, X_test, y_train, y_test = train_test_split(X_train_spec_act_sub_df, y_train_annotation, test_size=0.1)

    model_rf_spec_act_sub_df = simple_rf(random_state, bootstrap, n_estimators, max_depth, max_features, cores, X_train, y_train)
    prediction_test_spec_act_sub_df = model_rf_land_sub.predict(X_test)
    
    prediction_val_spec_act_sub_df = model_rf_spec_act_sub_df.predict(X_val_spec_act_sub_df)
    ccc_val_land_sub = get_ccc(y_val_annotation, prediction_val_spec_act_sub_df)
    
    # save model and prediction 
    model_name = s_path + 'omg_emp_model_rf_spec_act_sub_df.joblib.pkl'
    _ = joblib.dump(model_rf_spec_act_sub_df, model_name, compress=9)
    np.savetxt(s_path + 'prediction_val_spec_act_sub_df.csv', prediction_val_spec_act_sub_df, fmt='%.2f', delimiter=',')

    omg_emp_model_rf_land_sub = joblib.load(model_name)    
    
    
    # Stacked
    path_ = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/"
    prediction_val_spec_act_sub_df = pd.read_csv(path_ + "prediction_val_spec_act_sub_df.csv")
    
    
    # prediction on testset
    d_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/data/test/"
    m_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/"
    s_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/prediction_results/test/"
    
    # Landmarks (subjects)
    test_subs_landmarks = pd.read_csv(d_path + "test_subs_landmarks.csv", header = None)
    test_acts_landmarks = pd.read_csv(d_path + "test_acts_landmarks.csv", header = None)
    test_spec = np.load(d_path + "Test_spectogram.npy")
    test_deep_features = np.load(d_path + "deep_features_test.npy")
    
    model_subs = joblib.load(m_path + "omg_emp_model_rf_land_sub.joblib.pkl") 
    prediction_test_land_sub = model_subs.predict(test_subs_landmarks)
    
    np.savetxt(s_path + 'prediction_test_land_sub.csv', prediction_test_land_sub, fmt='%.2f', delimiter=',')
    
    # Landmarks (subjects + actors)
    model_acts_subs = joblib.load(m_path + "omg_emp_model_rf_land_act_sub.joblib.pkl") 
    prediction_test_land_act_sub = model_acts_subs.predict(np.hstack([test_acts_landmarks, test_subs_landmarks]))

    np.savetxt(s_path + 'prediction_test_land_act_sub.csv', prediction_test_land_act_sub, fmt='%.2f', delimiter=',')

    # spectogram, Landmarks (actors + subjects), deep features
    model_spec_act_sub_df = joblib.load(m_path + 'omg_emp_model_rf_spec_act_sub_df.joblib.pkl')
    prediction_test_spec_act_sub_df = model_spec_act_sub_df.predict(np.hstack([test_spec, test_acts_landmarks, test_subs_landmarks, test_deep_features]))

    np.savetxt(s_path + 'prediction_test_spec_act_sub_df.csv', prediction_test_spec_act_sub_df, fmt='%.2f', delimiter=',')

    
    # Fusion 
    s_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/prediction_results/validation/"
    cnn_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/data/validation/"

    prediction_val_spec_act_sub_df = pd.read_csv(s_path + "prediction_val_spec_act_sub_df.csv", header=None)
    val_CNN = pd.read_csv(cnn_path + "val_CNN.csv", header=None)
    
    
    fused_p = np.mean(np.hstack([prediction_val_spec_act_sub_df, val_CNN]), axis = 1)
    fused_prediction_val_spec_act_sub_df_CNN = fused_p.reshape(fused_p.shape[0], 1)
    
    np.savetxt(s_path + 'fused_prediction_val_spec_act_sub_df_CNN.csv', fused_prediction_val_spec_act_sub_df_CNN, fmt='%.2f', delimiter=',')

    
    t_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/prediction_results/test/"
    prediction_test_spec_act_sub_df = pd.read_csv(s_path + "prediction_test_spec_act_sub_df.csv", header=None)
    test_CNN = pd.read_csv(cnn_path + "test_CNN.csv", header=None)
    
    fused_p = np.mean(np.hstack([prediction_test_spec_act_sub_df, test_CNN]), axis = 1)
    fused_prediction_test_spec_act_sub_df_CNN = fused_p.reshape(fused_p.shape[0], 1)
    
    np.savetxt(s_path + 'fused_prediction_test_spec_act_sub_df_CNN.csv', fused_prediction_test_spec_act_sub_df_CNN, fmt='%.2f', delimiter=',')

    
    
    






