import os
import numpy as np
from matplotlib import pyplot as plt


def main():
    values_all = [
        dict(key='acl18-beta',
             variable=r'$\beta$',
             metric='MCC',
             ticks=['1e-4', '', '1e-2', '', '1e+0'],
             values=[(0.1848, 0.0300),
                     (0.1874, 0.0358),
                     (0.1894, 0.0335),
                     (0.1910, 0.0315),
                     (0.1839, 0.0396)]),
        dict(key='acl18-lambda',
             variable=r'$\lambda$',
             metric='MCC',
             ticks=['1e-3', '', '1e-1', '', '1e+0'],
             values=[(0.1713, 0.0288),
                     (0.1713, 0.0293),
                     (0.1745, 0.0356),
                     (0.1914, 0.0317),
                     (0.1910, 0.0315)]),
        dict(key='kdd17-beta',
             variable=r'$\beta$',
             metric='MCC',
             ticks=['1e-4', '', '1e-2', '', '1e+0'],
             values=[(0.0688, 0.0284),
                     (0.0687, 0.0316),
                     (0.0733, 0.0195),
                     (0.0655, 0.0329),
                     (0.0366, 0.0366)]),
        dict(key='kdd17-lambda',
             variable=r'$\lambda$',
             metric='MCC',
             ticks=['1e-3', '', '1e-1', '', '1e+0'],
             values=[(0.0258, 0.0182),
                     (0.0264, 0.0189),
                     (0.0282, 0.0169),
                     (0.0315, 0.0280),
                     (0.0733, 0.0195)]),
    ]

    for v_dict in values_all:
        key = v_dict['key']
        avg = np.array([e[0] for e in v_dict['values']])
        std = np.array([e[1] for e in v_dict['values']])

        plt.rc('font', size=12)
        plt.figure(figsize=(3.2, 2.8))
        plt.ylabel(v_dict['metric'])
        plt.xlabel('Value of ' + v_dict['variable'])
        plt.plot(avg, marker='o', markersize=8)
        plt.fill_between(np.arange(len(avg)), avg - std, avg + std, alpha=0.3)
        plt.xticks(list(range(len(avg))), v_dict['ticks'])

        if key.startswith('acl18'):
            plt.ylim([0.13, 0.24])
        elif key.startswith('kdd17'):
            plt.ylim([-0.01, 0.11])

        out_path = '../out-fig/ablation/{}.png'.format(key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
