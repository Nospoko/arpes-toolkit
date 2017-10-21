import numpy as np
from utils import data as ud
from utils import analysis as ua

def showcase(path):
    arpes = ud.read_arpes(path)

    data = arpes['data']
    ax0 = arpes['ax0']
    ax1 = arpes['ax1']

    ax0 = 16.89 - ax0

    extent = [ax1[0], ax1[-1], ax0[-1], ax0[0]]

    plt.imshow(data, extent = extent, aspect = 'auto')
    plt.show()

def main():
    path = 'data/ARPES0001.txt'
    arpes = ud.read_arpes(path)
    it = 450
    ua.fit_functions(arpes, it, True)

if __name__ == '__main__':
    main()
