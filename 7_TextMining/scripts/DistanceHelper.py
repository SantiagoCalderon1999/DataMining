from enum import Enum

class DistanceType(Enum):
    MANHATTAN = 1
    EUCLIDEAN = 2

import math

def _get_numeric_distance_euclidean(v_0: float, v_1: float):
    return (v_0 - v_1) ** 2

def _get_numeric_distance_manhattan(v_0: float, v_1: float):
    return abs(v_0 - v_1)

def _get_nominal_distance(v_0, v_1):
    return (float)(not v_0 == v_1)

def get_distance(x_0, x_1, length, type: DistanceType):
    if (len(x_0) != length or len(x_1) != length):
        raise Exception('Wrong vector size')
    
    total_sum = 0
    for i in range(0, length):
        current_distance = 0
        if (isinstance(x_0[i], str)):
            current_distance = _get_nominal_distance(x_0[i], x_1[i])
        else:
            if (type == DistanceType.EUCLIDEAN):
                current_distance = _get_numeric_distance_euclidean(x_0[i], x_1[i])
            else:
                current_distance = _get_numeric_distance_manhattan(x_0[i], x_1[i])
        total_sum += current_distance
    
    if (type == DistanceType.EUCLIDEAN):
        return math.sqrt(total_sum)
    else:
        return total_sum
    
    
def get_cosine_similarity(x_0, x_1, length):
    if (len(x_0) != length or len(x_1) != length):
        raise Exception('Wrong vector size')
    
    total_sum = 0
    for i in range(0, length):
        current_distance = 0
        if (isinstance(x_0[i], str)):
            current_distance = _get_nominal_distance(x_0[i], x_1[i])
        else:
            if (type == DistanceType.EUCLIDEAN):
                current_distance = _get_numeric_distance_euclidean(x_0[i], x_1[i])
            else:
                current_distance = _get_numeric_distance_manhattan(x_0[i], x_1[i])
        total_sum += current_distance
    
    if (type == DistanceType.EUCLIDEAN):
        return math.sqrt(total_sum)
    else:
        return total_sum