import numpy as np
import time

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity


def get_user_movie_score(data_path):
    user_movie_score = {}
    with open(data_path, 'r') as listFile:
        for line in listFile:
            user_id, movie_id, score, data_time = line.split()
            if user_id not in user_movie_score:
                user_movie_score[user_id] = [(movie_id, float(score))]
            else:
                user_movie_score[user_id].append((movie_id, float(score)))
    return user_movie_score


def get_score_matrix(user_movie_score, train_data_path, test_data_path):
    # 将user id与movie id映射到以0开始
    user_idx, movie_idx = {}, {}
    count_user, count_movie = 0, 0
    for key, values in user_movie_score.items():
        user_idx[key] = count_user
        for i in range(len(values)):
            movie_id, score = values[i]
            if movie_id not in movie_idx:
                movie_idx[movie_id] = count_movie
                count_movie += 1
        count_user += 1

    # 定义训练集，测试集用户评分矩阵，默认没有评分的位置为0
    train_score_matrix = np.zeros(shape=(count_user, count_movie))
    test_score_matrix = np.zeros(shape=(count_user, count_movie))
    with open(train_data_path, 'r') as listFile:
        for line in listFile:
            user_id, movie_id, score, data_time = line.split()
            row_idx = user_idx[user_id]
            column_idx = movie_idx[movie_id]

            # 将该用户对电影的分数填到对应的位置
            train_score_matrix[row_idx][column_idx] = float(score)
    
    with open(test_data_path, 'r') as listFile:
        for line in listFile:
            user_id, movie_id, score, data_time = line.split()
            row_idx = user_idx[user_id]
            column_idx = movie_idx[movie_id]
            test_score_matrix[row_idx][column_idx] = float(score)

    return train_score_matrix, test_score_matrix


def collaborative_filtering(matrix):
    # 计算每两个用户之间的余弦相似度，本质就是attn map
    sim_matrix = cosine_similarity(matrix)

    # 将相似度矩阵与原分数矩阵相乘，矩阵中的每一行为用户i对电影j的预测分数
    pred_score_matrix = np.dot(sim_matrix, matrix)

    # 再对每一行除以对应的相似度权重
    weight_sum_vec = sim_matrix.sum(axis=1)
    weight_sum_vec = weight_sum_vec[:, np.newaxis]
    weight_sum_vec = weight_sum_vec.repeat([pred_score_matrix.shape[1]], axis=1)
    pred_score_matrix = pred_score_matrix / weight_sum_vec
    return pred_score_matrix
 

def cal_RMSE(pred_score_matrix, test_score_matrix):
    """
    Args:
        pred_score_matrix ([numpy]): the prediction of score matrix
        test_score_matrix ([numpy]): test matrix label

    Returns:
        [float]: the RMSE value
    """
    pred = pred_score_matrix[test_score_matrix.nonzero()].flatten()
    label = test_score_matrix[test_score_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, label))


if __name__ == "__main__":
    train_data_path = "./data/netflix_train.txt"
    test_data_path = "./data/netflix_test.txt"

    # 获取训练集的user和对应的movie id, score
    train_user_movie_score = get_user_movie_score(data_path=train_data_path)
    train_score_matrix, test_score_matrix = get_score_matrix(user_movie_score=train_user_movie_score,
        train_data_path=train_data_path, test_data_path=test_data_path)

    start_time = time.time()
    pred_score_matrix = collaborative_filtering(train_score_matrix)
    algorithm_time = time.time() - start_time
    rmse = cal_RMSE(pred_score_matrix, test_score_matrix)
    print("RMSE: {}".format(rmse))
    print("Collaborative Filtering algorithm time: {}".format(algorithm_time))

