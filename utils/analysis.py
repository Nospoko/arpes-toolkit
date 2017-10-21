import numpy as np
from tqdm import tqdm
from scipy import optimize as so
from matplotlib import pyplot as plt

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

    bounds = [[-np.inf, 10, -np.inf, -np.inf],
              [np.inf, 10000, np.inf, np.inf]]
    score = so.least_squares(lorentz_fit, params, bounds = bounds)

    return k, score

def fit_arpes(k, arpes_row):
    # Function to actually fit
    def lorentz_fit(params):
        fit = lorentzian(k, params)
        return (fit - arpes_row)

    # params = y0, A, w, xc
    params = 0, 1000, 1, -0.1
    bounds = [[-np.inf, 10, -np.inf, -np.inf],
              [np.inf, 10000, np.inf, np.inf]]
    score = so.least_squares(lorentz_fit, params, bounds = bounds)

    return score

def analyse_arpes(arpes, Eb_span = [0.25, 0.5]):
    # Prepare energy axis
    Ekins = arpes['ax0']
    Eb = 16.89 - arpes['ax0']

    angle = arpes['ax1']

    # Select the range
    Eb_min, Eb_max = Eb_span
    ids = (Eb > Eb_min) & (Eb < Eb_max)
    ids = np.where(ids)[0]

    scores = []
    for it in ids:
        Ekin = Ekins[it]
        k = transform_angle(angle, Ekin)
        arpes_row = arpes['data'][it][:-1]
	score = fit_arpes(k, arpes_row)
	scores.append(score)

    # Extract the lorentzian width parameter
    w1 = []
    for score in scores:
        w1.append(score.x[2])

    # And the corresponding energy values
    energies = Eb[ids]

    return energies, w1
