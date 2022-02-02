import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def signal_simulation(a, sigm, w, start, end, N, noise=0.01, plot=False):
    """
    input:

    t - вектор моментов, когда производился замер интенсивностей

    a - Матрица средних функций распределений, строки соотвествуют сигналам
    sigm - вектор отклонений функций распределений
    w - вектор весов, с которыми берутся ф.р.
    start, end: float, граничные точки отрезка
    N: int, число точек
    noise: float отклонение шума с распределением N(0, noise^2)
    plot: bool, выводить ли картинку
    output:
    зашумленный вектор интеснивностей
    """
    a = np.array(a)
    sigm = np.array(sigm)
    w = np.array(w)

    t = np.linspace(start, end, N)
    f = np.zeros((a.shape[0], N))

    for j in range(a.shape[0]):
        for i in range(a.shape[1]):
            f[j] += w[j, i] * sts.norm.pdf(t, loc=a[j, i], scale=sigm[j, i])
        # добавим шум
        f[j] += sts.norm.rvs(loc=0, scale=noise, size=N)

    if np.any(f < 0):
        f -= np.min(f)

    # центрируем
    f -= np.mean(f, axis=0)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        for j in range(a.shape[0]):
            ax.plot(t, f[j], label='Сигнал' + str(j + 1))

        pca = PCA()
        # В сислу большой размерности пространства компоненты - число сигналов
        pca.fit(f)
        f_components = pca.components_

        xevents1 = EventCollection((t[-1] / np.max(f_components[0])) * f_components[0], color='tab:blue',
                                   linelength=0.05, label='First component')
        xevents2 = EventCollection((t[-1] / np.max(f_components[1])) * f_components[1], color='tab:orange',
                                   linelength=0.05, label='Second component')
        #  yevents1 = EventCollection((np.max(f)/np.max(f_components[0]))*f_components[0], color='tab:blue', linelength=0.05, orientation='vertical')
        #  yevents2 = EventCollection((np.max(f)/np.max(f_components[1]))*f_components[1], color='tab:orange', linelength=0.05, orientation='vertical')

        ax.add_collection(xevents1)
        ax.add_collection(xevents2)
        #   ax.add_collection(yevents1)
        #   ax.add_collection(yevents2)

        ax.set_xlabel('Частота')
        ax.set_ylabel('Сигнал')
        ax.legend()
        plt.show()

    return f