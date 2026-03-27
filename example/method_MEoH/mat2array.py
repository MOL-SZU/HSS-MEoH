import scipy.io
import numpy as np


def load_mat_to_numpy(file_path, variable_name=None):
    """
    加载 .mat 文件并转换为 numpy array

    参数:
    file_path: .mat 文件路径
    variable_name: 指定要提取的变量名，如果为None则提取所有变量
    """
    # 加载 .mat 文件
    mat_data = scipy.io.loadmat(file_path)

    # 如果指定了变量名
    if variable_name is not None:
        if variable_name in mat_data:
            data = mat_data[variable_name]
            return data
        else:
            available_vars = [key for key in mat_data.keys() if not key.startswith('__')]
            raise KeyError(f"变量 '{variable_name}' 不存在。可用变量: {available_vars}")

    # 如果没有指定变量名，返回所有数据变量
    else:
        result = {}
        for key in mat_data.keys():
            if not key.startswith('__'):
                result[key] = mat_data[key]
        return result
