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

def showcase(path):
    arpes = read_arpes(path)

    data = arpes['data']
    ax0 = arpes['ax0']
    ax1 = arpes['ax1']

    ax0 = 16.89 - ax0

    extent = [ax1[0], ax1[-1], ax0[-1], ax0[0]]

    plt.imshow(data, extent = extent, aspect = 'auto')
    plt.show()

def double_gaussian(x, params):
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) 
    return res

def lorentzian(x, params):
    y0, A, w, xc = params
    res = y0 + (2*A / np.pi) * w / (4 * (x - xc) ** 2 + w ** 2)

    return res

def double_lorentzian(x, params):
    y01, A1, w1, xc1, y02, A2, w2, xc2 = params
    params1 = y01, A1, w1, xc1
    params2 = y02, A2, w2, xc2
    # (c1, pos1, gm1, c2, pos2, gm2, off) = params
    l1 = lorentzian(x, params1)
    l2 = lorentzian(x, params2)
    
    res = l1 + l2

    return res

def transform_angle(angle, Ekin):
    # Electron mass [eV / c^2]
    c = 299792458.
    me = 1e6 * 0.511 / c ** 2

    # Planck constant barred [eV * s]
    hbar = 1e-16 * 6.58

    # Data is in degrees
    th = angle * np.pi / 180.

    # Wave vector length [1/m]
    k = ( 1. / hbar ) * np.sqrt(2. * me * Ekin) * np.sin(th)

    # Rescale to [1 / A]
    k /= 1e10

    return k

def main():
    path = 'data/ARPES0001.txt'
    arpes = read_arpes(path)
    it = 450
    fit_functions(arpes, it, True)

def fit_function(arpes, it):
    data = arpes['data']

    # There is a discrepency in the number of samples
    Ekins = arpes['ax0']
    Ekin = Ekins[it]
    Eb = 16.89 - arpes['ax0']

    energy = Eb[it]
    angle = arpes['ax1']

    row = data[it][:-1]
    k = transform_angle(angle, Ekin)

    def lorentz_fit(params):
        fit = lorentzian(k, params)
        return (fit - row)

    # params = y0, A, w, xc
    params = 0, 1000, 1, -0.1

    score = so.least_squares(lorentz_fit, params)

    return k, score

def fit_functions(arpes, it, show = False):
    data = arpes['data']

    # There is a discrepency in the number of samples
    Ekins = arpes['ax0']
    Ekin = Ekins[it]
    Eb = 16.89 - arpes['ax0']

    energy = Eb[it]
    angle = arpes['ax1']

    row = data[it][:-1]
    k = transform_angle(angle, Ekin)

    # def double_gaussian_fit(params):
    #     fit = double_gaussian(x, params )
    #     return (fit - row)
    #
    # score = so.least_squares(double_gaussian_fit, [1000.0, 13.0, 1.0, 1000.0, 1.0, 1.0])

    def double_lorentz_fit(params):
        fit = double_lorentzian(k, params)
        return (fit - row)

    # params = y0, A, w, xc
    params1 = 0, 1000, 1, -0.1
    params2 = 0, 1000, 1, 0.1
    params = params1 + params2

    score = so.least_squares(double_lorentz_fit, params)

    params1 = score.x[:4].copy()
    params2 = score.x[4:].copy()
    ymean = params1[0] + params2[0]
    ymean *= 0.5
    params1[0] = ymean
    params2[0] = ymean

    if show:
        plt.plot(k, row)
        plt.plot(k, double_lorentzian(k, score.x), c='r')
        plt.plot(k, lorentzian(k, params1), '--')
        plt.plot(k, lorentzian(k, params2), '--')
        plt.title('En = {}'.format(energy))
        plt.show()

    return score

if __name__ == '__main__':
    showcase()
