import numpy as np

def read_arpes(path):
    with open(path) as fin:
         lines = fin.readlines()

    results = {}
    queries = ['Dimension 1 scale',
               'Dimension 2 scale',
               'Dimension 1 name',
               'Dimension 2 name',
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


