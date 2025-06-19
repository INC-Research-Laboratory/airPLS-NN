import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import tensorflow as tf
from tensorflow.keras import backend as K
import time
import pickle
import random
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def WhittakerSmooth(x,w,lambda_,differences=1):
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1]
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, data_len, lambda_, porder, itermax, dssn_criterion):
    m=x.shape[0]
    w=np.ones(m)
    z = np.empty(data_len,)

    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<dssn_criterion*(abs(x)).sum() or i==itermax):
            if(i==itermax): pass
            break
        w[d>=0]=0
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn)
        w[-1]=w[0]
    return z

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def SMAPE(y_true, y_pred):
    return K.mean((K.abs(y_true-y_pred))/(K.abs(y_true)+K.abs(y_pred)))*100

def y_return_per_column(train_df, valid_df, test_df, obj, column_name):
    train_y_df = obj.fit_transform(train_df[[column_name]])
    valid_y_df = obj.transform(valid_df[[column_name]])
    test_y_df = obj.transform(test_df[[column_name]])
    return train_y_df, valid_y_df, test_y_df

def compare_graph_plot(data_len, correction_file, original_file, lambda_original_value_list, itermax_original_value_list, dssn_criterion_original_value_list, porder_original_value_list):
    lambda_ = lambda_original_value_list[0]
    lambda_ = int(lambda_)
    itermax = itermax_original_value_list[0]
    itermax = int(itermax)
    dssn_criterion = dssn_criterion_original_value_list[0]
    porder = porder_original_value_list[0]
    porder = int(round(float(porder)))

    correction_df = pd.read_csv(correction_file, header=None)
    data = correction_df.values
    smooth_drift_data = np.concatenate(data).tolist()
    plt.plot(smooth_drift_data, label='Baseline-added data')

    smooth_drift_data=np.array(smooth_drift_data)
    correction = smooth_drift_data-airPLS(smooth_drift_data, data_len, lambda_=lambda_, porder=porder, itermax=itermax, dssn_criterion=dssn_criterion)
    plt.plot(correction, label='Corrected data')

    original_df = pd.read_csv(original_file, header=None)
    data = original_df.values
    original = np.concatenate(data).tolist()
    plt.plot(original, label='Original data')

    plt.legend()

def rescaling_parameter(predict, column_name):
    original_value_list = []

    predict = pd.DataFrame(predict, columns=[column_name])

    if column_name == 'lambda_':
        with open('./model/lambda_.pkl', 'rb') as f:
            lambda__scaler = pickle.load(f)
        if predict[[column_name]].values.item() >= 1.0:
            original_value = float(10 ** math.trunc(9.0*10))
        else:
            original_value = float(10 ** round(predict[[column_name]].values.item() * 10))
        original_value_list.append(original_value)
        original_value_list = np.array(original_value_list)
        return original_value_list

    if column_name == 'itermax':
        with open('./model/itermax.pkl', 'rb') as f:
            itermax_scaler = pickle.load(f)
        original_value = itermax_scaler.inverse_transform(predict[[column_name]])
    elif column_name == 'dssn_criterion':
        with open('./model/dssn_criterion.pkl', 'rb') as f:
            dssn_criterion_scaler = pickle.load(f)
        original_value = dssn_criterion_scaler.inverse_transform(predict[[column_name]])
        original_values = [0.000001, 0.00001, 0.00005, 0.0001, 0.0005]
        original_value = original_values[round(original_value[0][0])]
    elif column_name == 'porder':
        with open('./model/porder.pkl', 'rb') as f:
            porder_scaler = pickle.load(f)
        original_value = porder_scaler.inverse_transform(predict[[column_name]])
    original_value_list.append(original_value)
    return original_value_list