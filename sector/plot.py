import matplotlib.pyplot as plt
import numpy as np

AUXILIARY_LOSS_XS = [f'r{round(i * 0.1, 2)}' for i in range(16)]

def draw_line_chart(x, ys, legends, path = 'dd.png'):
    plt.clf()
    for y, l in zip(ys, legends):
        plt.plot(x[:len(y)], y, label = l)
    plt.legend()
    plt.savefig(path)


def test_draw_line_chart():
    x = ['r00', 'r01','r02','r03','r04']
    ys = [[10, 11, 12, 13, 12], [9, 12, 13, 10, 8]]
    legends = ['dd1', 'dd2']
    draw_line_chart(x, ys, legends)
