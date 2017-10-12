import numpy as np
import igor.igorpy as igor
from scipy import optimize as so
from matplotlib import pyplot as plt

def read_arpes(path):
    with open(path) as fin:
         lines = fin.readlines()

    results = {}
    queries = ['Dimension 1 scale',
               'Dimension 2 scale',
               'Dimension 1 name',
               'Dimension 1 name',
               'Excitation Energy',
               'Spectrum Name']

    data = []
    read_data = False
    for line in lines[:-1]:
        if not read_data:
            for query in queries:
                if query in line:
                    out = line.split('=')[1].strip()
                    results[query] = out
            if '[Data 1]' in line:
                read_data = True
        else:
            line = line.strip()
            row = line.split('  ')
            row = [np.float(val) for val in row]
            data.append(row)

    # Convert numeric data properly
    results['data'] = np.array(data)
    ax0 = [float(val) for val in results[queries[0]].split(' ')]
    ax1 = [float(val) for val in results[queries[1]].split(' ')]
    results['ax0'] = np.array(ax0)
    results['ax1'] = np.array(ax1)

    return results

def showcase():
    path = 'data/ARPES0001.txt'
    arpes = read_arpes(path)

    data = arpes['data']
    ax0 = arpes['ax0']
    ax1 = arpes['ax1']

    ax0 = 16.89 - ax0

    extent = [ax1[0], ax1[-1], ax0[-1], ax0[0]]

    plt.imshow(data, extent = extent, aspect = 'auto')
    plt.show()

def double_gaussian(x, params):
    (c1, mu1, sigma1, c2, mu2, sigma2, off) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
          + off
    return res

def fit_gausians():
    path = 'data/ARPES0001.txt'
    arpes = read_arpes(path)
    data = arpes['data']

    row = data[300]
    x = np.arange(len(row))

    def double_gaussian_fit(params):
        fit = double_gaussian(x, params )
        return (fit - row)

    fit = so.least_squares(double_gaussian_fit, [100.0, 13.0, 1.0, 100.0, 123.0, 1.0, 10])

    # return fit

    plt.plot(x, row)
    plt.plot(x, double_gaussian(x, fit.x), c='r')
    plt.show()

if __name__ == '__main__':
    showcase()
