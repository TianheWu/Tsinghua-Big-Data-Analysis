import numpy as np







def get_movie_id(movie_data_path):
    movie_id_list = []
    with open(movie_data_path, 'r', encoding='ISO-8859-1') as listFile:
        for line in listFile:
            movie_id, year, title = line.split(",", 2)
            movie_id_list.append(movie_id)
    return np.array(movie_id_list)


def get_user_id(user_data_path):
    user_id_list = []
    with open(user_data_path, 'r') as listFile:
        for line in listFile:
            user_id = line[:-1]
            user_id_list.append(user_id)
    return np.array(user_id_list)


def read_train_file(train_data_path):
    user_movie_score = []
    with open(train_data_path, 'r') as listFile:
        for line in listFile:
            user_id, moive_id, score, data_time = line.split()
            user_movie_score.append([user_id, moive_id, score])
            

def get_matrix(user_num, movie_num):
    score_matrix = np.zeros(shape=(user_num, movie_num))
    


if __name__ == "__main__":
    user_data_path = "./data/users.txt"
    train_data_path = "./data/netflix_train.txt"
    movie_data_path = "./data/movie_titles.txt"
    

    movie_id_list = get_movie_id(movie_data_path)
    user_id_list = get_user_id(user_data_path)
    
    user_num = user_id_list.shape[0]
    movie_num = movie_id_list.shape[0]

    print(user_num)
    print(movie_num)
