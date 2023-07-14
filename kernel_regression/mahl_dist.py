#import numpy as np
#from numpy import linalg as LA
#from numpy import random
from cupy import linalg as LA
from cupy import random
import cupy as np
import time

#import tensorflow as tf


# 参考：https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/ReadMe.md
def cal_mahal_dist(query_matrix, matrix):
    matrix_center = np.mean(matrix, axis=0)
    delta = query_matrix - matrix_center

    # calculate the covariance matrix and its inverse matrix
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    cov_matrix_inv = LA.inv(cov_matrix)

    # calculate the Mahalanobis distance between a single vector and the center of the dataset
    def md_vector(vector):
        inner_prod = np.dot(vector, cov_matrix_inv)
        dist = np.sqrt(np.dot(inner_prod, vector))
        return dist

    mahal_dist = np.apply_along_axis(arr=delta, axis=1, func1d=md_vector)
    assert len(mahal_dist) == len(query_matrix)
    return mahal_dist


#参考:https://github.com/duozhanggithub/Mahalanobis-Distance/blob/master/Mahalanobis_Distance.ipynb
def cal_mahal_dist_pp(query_matrix, matrix, cov_matrix_inv):  # 点到点的马氏距离

    # calculate the covariance matrix and its inverse matrix
    #cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    #cov_matrix_inv = LA.inv(cov_matrix)

    qrow, qcol = query_matrix.shape
    row, col = matrix.shape

    assert qcol == col

    #print('begin to calc diff matrix')
    #t1 = time.time()
    diff_mat = np.expand_dims(query_matrix, 1) - matrix
    #t2 = time.time()
    #print('calc diff matrix cost time: {}'.format(t2-t1))

    #print('diff_mat shape:', diff_mat.shape)

    inner_prod = np.matmul(diff_mat, cov_matrix_inv)
    #inner_prod = tf.linalg.matmul(diff_mat_tensor, cov_matrix_tensor)

    #t3 = time.time()

    #print('inner_prod shape', inner_prod.shape)

    #print('calc inner prod cost time: {}'.format(t3-t2))

    inner_prod = np.reshape(inner_prod, (qrow*row, col))
    #inner_prod = tf.reshape(inner_prod, (qcol*col, row))

    diff = np.reshape(diff_mat, (qrow*row, col))

    mahalanobis_distance = np.sqrt(np.diag(np.dot(inner_prod, diff.T)))

    mahalanobis_distance = np.reshape(mahalanobis_distance, (qrow, row))

    return mahalanobis_distance



query_row = 50000
query_mat = random.random((query_row,3072))
dataset_mat = random.random((10000,3072))
#dist = cal_mahal_dist(query_mat, dataset_mat) # 计算点到分布的马氏距离
#print(dist.shape)

# calculate the covariance matrix and its inverse matrix
cov_matrix = np.cov(dataset_mat, rowvar=False, ddof=1)
cov_matrix_inv = LA.inv(cov_matrix)

batch_size = 4
epco = query_row//batch_size+1

dists = []
for i in range(epco):
    query = query_mat[i*batch_size:(i+1)*batch_size]
    dist = cal_mahal_dist_pp(query, dataset_mat, cov_matrix_inv)
    dists.append(dist)
    print('dist shape', dist.shape)

dists = np.concatenate(dists, axis=0)
print('dists shape', dists.shape)
