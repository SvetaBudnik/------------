import math
import statistics
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import numpy as np

from scipy import stats as st
from scipy import integrate
from scipy.integrate import quad
from scipy.optimize import minimize

import pandas as pd


# Параметры для работы

kol_rand = 1000000  # количество которое хочешь сгенерить
# kol_rand = 500 # количество которое хочешь сгенерить
eps = 0.4  # Эпсилон
rng = np.random.default_rng(
    12345
)  # Генератор рандомных чисел с фиксированным сидом (для повторяемости результатов)

# Параметры для чистого распределения

N_net = 4  # Параметр распределения
l_net = 1  # Масштаб (лямбда)
O_net = 0  # Смещение (Тетта)
disp_net = 2 * N_net  # Дисперсия
excesses_net = 3 / N_net  # Эксцесс

# Параметры для шума

N_scattered = 4  # Параметр распределения
l_scattered = 2  # Масштаб (лямбда)
O_scattered = 0  # Смещение (Тетта)
disp_scattered = 2 * N_scattered * l_scattered**2  # Дисперсия
excesses_scattered = 3 / N_scattered  # Эксцесс


def gen_rand_laplas(n_lap: int, theta: float = 0.0, lamb: float = 1.0) -> float:
    """
    Генератор случайной величины, соответствующей обобщенному распределению Лапласа, с возможностью задания сдвига и масштабирования

    Args:
        n_lap (int): Натуральное число, соответствющее количеству случайных величин. Является параметром обобщённого распределения Лапласа `n`
        theta (float, optional): параметр сдвига распределения тэтта. По умолчанию 0.0
        lamb (float, optional): параметр масштабироваиня сдвига лямбда. По умолчанию 1.0

    Returns:
        float: Случайная величина, соответствующая обобщенному распределению Лапласа.
    """
    global rng

    list_rand = rng.random((n_lap,))

    res = 1.0
    for item in list_rand:
        if item <= (1 / 2):
            res *= 2 * item
        else:
            res /= 2 * (1 - item)

    return np.log(res) * lamb + theta


def gen_full_laplas(
    count: int, theta: float = 0.0, lamb: float = 1.0
) -> np.ndarray[np.float64]:
    """Генератор множества случайных величин по распределению Лапласа

    Args:
        count (int): размер выборки случайных чисел по данному распределению
        theta (float, optional): параметр сдвига распределения тэтта. По умолчанию 0.0
        lamb (float, optional): параметр масштабироваиня сдвига лямбда. По умолчанию 1.0

    Returns:
        ndarray[float64]: множество значений полученной выборки
    """
    global N_net

    res = [gen_rand_laplas(N_net, theta=theta, lamb=lamb) for _ in range(count)]
    return np.asarray(res)


def get_scattened_laplas(
    clean: np.ndarray[np.float64], scattener: np.ndarray[np.float64], eps: float
) -> np.ndarray[np.float64]:
    """Соединяет чистое и загрязняющее распределения в единое загрязнённое

    Args:
        clean (np.ndarray[np.float64]): множество случайных величин из чистого распределения
        scattener (np.ndarray[np.float64]): множество случайных величин из загрязняющего распределения (должно совпадать по размеру с чистым)
        eps (float): коэффициент загрязнения (0 <= `eps` <= 0.5)

    Returns:
        np.ndarray[np.float64]: полученное загрязнённое распределение
    """
    global rng
    res = np.zeros_like(clean)
    rand = rng.random((clean.size,))

    for ind in range(res.size):
        if rand[ind] <= (1 - eps):
            res[ind] = clean[ind]
        else:
            res[ind] = scattener[ind]

    return res


def ploat_graphics(
    clean: np.ndarray[np.float64],
    scattener: np.ndarray[np.float64],
    scattened: np.ndarray[np.float64],
):
    """Выводит графики сгенерированных распределений на экран (приводя размеры к единичному по графику `clean`)

    Args:
        clean (np.ndarray[np.float64]): массив значений из чистого распределения
        scattener (np.ndarray[np.float64]): массив значений из загрязняющего распределения
        scattened (np.ndarray[np.float64]): массив значений из грязного распределения
    """
    # Сгруппировать данные и просуммировать их по группам
    count_of_groups = 125
    lowest_value = min(clean.min(), scattener.min(), scattened.min())
    highest_value = max(clean.max(), scattener.max(), scattened.max()) + 1

    x_values = np.linspace(lowest_value, highest_value, count_of_groups)
    clean_y = np.zeros_like(x_values)
    scattener_y = np.zeros_like(x_values)
    scattened_y = np.zeros_like(x_values)

    clean.sort()
    scattener.sort()
    scattened.sort()

    # ОПАСНО! Такая реализация может добавлять неверные значения при малой выборке и большом числе групп!!
    i = 1
    for el in clean:
        if el <= x_values[i]:
            clean_y[i - 1] += 1
        else:
            i += 1
            clean_y[i - 1] += 1

    i = 1
    for el in scattener:
        if el <= x_values[i]:
            scattener_y[i - 1] += 1
        else:
            i += 1
            scattener_y[i - 1] += 1

    i = 1
    for el in scattened:
        if el <= x_values[i]:
            scattened_y[i - 1] += 1
        else:
            i += 1
            scattened_y[i - 1] += 1

    # Провести масштабирование графиков
    max_y = clean_y.max()
    clean_y = clean_y / max_y
    scattener_y = scattener_y / max_y
    scattened_y = scattened_y / max_y

    # Интерполируем графики в сплайны
    clean_spline = make_interp_spline(x_values, clean_y)
    scattener_spline = make_interp_spline(x_values, scattener_y)
    scattened_spline = make_interp_spline(x_values, scattened_y)

    X_ = np.linspace(x_values.min(), x_values.max(), 1000)

    cl_smooth = clean_spline(X_)
    scer_smooth = scattener_spline(X_)
    sced_smooth = scattened_spline(X_)

    # Выводим полученные графики
    fig, axes = plt.subplots()
    axes.plot(X_, cl_smooth, label="чистое", linestyle="--", color="red")
    axes.plot(X_, scer_smooth, label="загрязняющее", linestyle="-.", color="blue")
    axes.plot(X_, sced_smooth, label="загрязнённое", linestyle="-", color="green")
    axes.plot([0, 0], [0.0, 1.1], label="Центр", color="purple")
    axes.legend()

    plt.show()


# вычисление выборочных характеристик: среднего арифметического, выборочной медианы, дисперсии, коэффициентов асимметрии и эксцесса;

size = 800000

print(f"Размер выборки: {size} элементов...")

print("Строим чистое распределение...")
clean = gen_full_laplas(size)

print("Строим загрязняющее распределение...")
scattener = gen_full_laplas(size, theta=10)

print("Строим загрязнённое распределение...")
scattened = get_scattened_laplas(clean, scattener, 0.2)

print("Делаем графики...")
ploat_graphics(clean, scattener, scattened)

