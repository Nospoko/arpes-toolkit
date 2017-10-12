import numpy as np
import igor.igorpy as igor
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

    results['data'] = np.array(data)

    return results

def showcase():
    path = 'data/ARPES0001.pxt'
    data = igor.load(path)

    ax0 = data.children[0].axis[0]
    ax1 = data.children[0].axis[1]

    arpes = data.children[0].data

    plt.imshow(arpes)
    plt.show()

if __name__ == '__main__':
    showcase()
