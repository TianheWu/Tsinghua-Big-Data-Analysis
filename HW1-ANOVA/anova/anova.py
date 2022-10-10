import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, normaltest, shapiro


def log_transform_func(x):
    return np.log(x)

def cal_sample_variance(x, mean):
    sum = 0
    for _x in x:
        sum += (_x - mean) ** 2
    sum /= (x.shape[0] - 1)
    return sum

def cal_sample_std(x):
    return np.std(x, ddof=1)

def cal_mean(x):
    return np.mean(x)

def cal_ssb(x):
    grand_mean = 0
    n = 0
    ssb = 0
    for key, value in x.items():
        n += value.shape[0]
        grand_mean += value.shape[0] * cal_mean(value)
    grand_mean /= n
    for key, value in x.items():
        n_group = value.shape[0]
        ssb += n_group * (cal_mean(value) - grand_mean) ** 2
    return ssb

def cal_ssw(x):
    ssw = 0
    for key, value in x.items():
        mean_group = cal_mean(value)
        for v in value:
            ssw += (v - mean_group) ** 2
    return ssw

def get_between_df(x):
    n_gropus = len(x.keys())
    return n_gropus - 1

def get_within_df(x):
    df_w = 0
    for key, value in x.items():
        n_group = value.shape[0]
        df_w += n_group - 1
    return df_w

def one_way_anova(x):
    ssb = cal_ssb(x)
    ssw = cal_ssw(x)
    total = ssb + ssw
    df_b = get_between_df(x)
    df_w = get_within_df(x)
    df_total = df_b + df_w
    ms_b = ssb / df_b
    ms_w = ssw / df_w
    F = ms_b / ms_w
    print("Source ==== SS ==== df ==== MS ==== F ==== D")
    print("Between === {} ==== {} ==== {} ==== {} ==== {}".format(ssb, df_b, ms_b, F, 0))
    print("Within === {} ==== {} ==== {} ==== {} ==== {}".format(ssw, df_w, ms_w, 0, 0))
    print("Total === {} ==== {} ==== {} ==== {} ==== {}".format(total, df_total, 0, 0, 0))


def split_data_from_category(categories_list, data_list):
    ret_dict = {}
    temp_list = []
    pre_category = 1
    for i in range(categories_list.shape[0]):
        if categories_list[i] != pre_category:
            ret_dict[pre_category] = np.array(temp_list)
            pre_category = categories_list[i]
            temp_list = []
        temp_list.append(data_list[i])
    ret_dict[pre_category] = np.array(temp_list)
    return ret_dict


def test_normality(data):
    data = pd.DataFrame(data, columns=['value'])
    u = data['value'].mean()
    std = data['value'].std()  # 计算标准差
    print('The result of Kolmogorov-Smirnov (K-S) test:')
    print(kstest(data['value'], 'norm', (u, std)))
    print('The result of normaltest test:')
    print(normaltest(data['value']))
    print('The result of Shapiro-Wilk (S-W) test:')
    print(shapiro(data['value']))


def visual_hist_pdf(data, save_fig_path, col_name):
    data = pd.DataFrame(data, columns=[col_name])
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(data.index, data.values)
    plt.grid()
    ax2 = fig.add_subplot(2, 1, 2)
    data.hist(bins=30, alpha=0.5, ax=ax2)
    data.plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid()
    plt.savefig(save_fig_path)
    plt.show()


def get_column_from_xlsx(file_path, usecols=[1]):
    ret_list = []
    data = pd.read_excel(io=file_path, sheet_name='data', usecols=usecols)
    for i in range(data.shape[0]):
        ret_list.append(data.iloc[i, 0])
    return np.array(ret_list)


def anova_assumptions(file_path, save_fig_path, col, col_name, categories, log_transform=False):
    print("col {}, col_name {} ======================================".format(col + 1, col_name))
    data = get_column_from_xlsx(file_path=file_path, usecols=[col])
    if log_transform:
        data = data + 1e-12
        data = log_transform_func(data)
        point, figure, name = save_fig_path.split('/')
        name = "log-" + name
        save_fig_path = point + '/' + figure + '/' + name
        col_name = "log-" + col_name
    categories_data_dict = split_data_from_category(categories, data)
    visual_hist_pdf(data, save_fig_path=save_fig_path, col_name=col_name)

    # Normality
    test_normality(data)

    # Homogeneity of variance
    for key, value in categories_data_dict.items():
        mean = cal_mean(value)
        std = cal_sample_std(value)
        var = cal_sample_variance(value, mean=mean)
        print("Category {}| num: {}, mean: {}, std: {}, var: {}".format(key, value.shape[0], mean, std, var))


if __name__ == "__main__":
    file_path = "./data.xlsx"
    col_name = ["Group Name", "Group Category", "Group Size", "Message Number", "Friendship Relational Density",
                "Sex Ration", "Average Age", "Variance of Age", "Geographical Area", "Mobile Conversation Ratio", 
                "Conversation Number", "No-response Conversation Ratio", "Night Conversation Ratio", "Images Ratio"]    
    categories = get_column_from_xlsx(file_path=file_path, usecols=[1])
    # for i in range(2, 14):
    #     save_fig_path = "./figure/" + "Col-" + str(i + 1) + "_" + col_name[i].replace(" ", "_") + "_hist_pdf.jpg"
    #     anova_assumptions(file_path=file_path, save_fig_path=save_fig_path, col=i,
    #         col_name=col_name[i], categories=categories, log_transform=True)
    

    data = get_column_from_xlsx(file_path=file_path, usecols=[12])
    categories_data_dict = split_data_from_category(categories, data)
    one_way_anova(categories_data_dict)
    