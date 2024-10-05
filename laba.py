import math
import statistics
import matplotlib.pyplot as plt

import numpy as np

from scipy import stats as st
from scipy import integrate
from scipy.integrate import quad
from scipy.optimize import minimize



# Параметры для работы

kol_rand = 1000000 # количество которое хочешь сгенерить
# kol_rand = 500 # количество которое хочешь сгенерить
eps = 0.4 # Эпсилон

# Параметры для чистого распределения

N_net = 2 # Параметр распределения
l_net = 1 # Масштаб (лямбда)
O_net = 0 # Смещение (Тетта)
disp_net = 2 * N_net # Дисперсия
excesses_net = 3 / N_net # Эксцесс

# Параметры для шума

N_scattered = 2 # Параметр распределения
l_scattered = 2 # Масштаб (лямбда)
O_scattered = 4 # Смещение (Тетта)
disp_scattered = 2 * N_scattered * l_scattered**2 # Дисперсия
excesses_scattered = 3/N_scattered # Эксцесс


def gen_rand_laplas(n_lap: int) -> float:
    """
    Генератор случайной величины, соответствующей обобщенному распределению Лапласа.

    Параметры
    ----------
    n_lap : int
        Натуральное число, соответствющее количеству случайных величин.

    Возвращает
    ----------
    out : float
        Случайная величина, соответствующая обобщенному распределению Лапласа.
    """
    list_rand = np.random.rand(n_lap)
    top_mult = 1
    down_mult = 1
    for item in list_rand:
        if item <= 1/2:
            top_mult = top_mult * 2 * item
        elif item > 1/2:
            down_mult = down_mult * 2 * (1 - item)
    return np.log(top_mult / down_mult)


def mix_gen_rand_laplas(n_net: int, n_scattered: int, l: float, O: float) -> float:
    """
    Генератор случайной величины соответствующей обобщенному смешанному распределению Лапласа.

    Параметры
    ----------
    n_net : int
        Количество случайных величин в чистом распредении.
    n_scattered : int
        Количество случайных величин в загрязненном распредении.
    l : float
        Масштаб распределения.
    O : float
        Смещение распределения.

    Возвращает
    ----------
    out : float
        Случайная величина, соответствующая обобщенному обобщенному смешанному распределению Лапласу.
    """
    rand_eps = np.random.rand(1)
    top_mult = 1
    down_mult = 1
    if rand_eps < 1 - eps:
        list_rand = np.random.rand(n_net)
        for item in list_rand:
            if item <= 1/2:
                top_mult = top_mult * 2 * item
            elif item > 1/2:
                down_mult = down_mult * 2 * (1 - item)
        return np.log(top_mult/down_mult)
    else:
        list_rand = np.random.rand(n_scattered)
        for item in list_rand:
            if item <= 1/2:
                top_mult = top_mult * 2 * item
            elif item > 1/2:
                down_mult = down_mult * 2 * (1 - item)
        # return (np.log(top_mult/down_mult) - O)/l
        # return (np.log(top_mult/down_mult) + O) * l
        return np.log(top_mult/down_mult) * l + O

# вычисление выборочных характеристик: среднего арифметического, выборочной медианы, дисперсии, коэффициентов асимметрии и эксцесса;




import pandas as pd