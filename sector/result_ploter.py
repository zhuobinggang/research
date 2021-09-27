# import result_grid2 as R2
# import result_090521 as R1
# import result_091321 as R3
import result_092721 as R4
from plot import *
import seaborn as sns # for heatmap

# 0905
def draw():
    y2 = R2.get_e2_avgs(R2.rates, 16)
    y1 = R1.get_e2_avgs()
    draw_line_chart(AUXILIARY_LOSS_XS, [y1, y2], ['2nd', '1st'])

def output_heatmap(mat, xs, ys, path = 'dd.png'):
    plt.clf()
    sns.heatmap(mat, xticklabels=xs, yticklabels=ys, cmap='Blues', annot=True, fmt='.4g')
    plt.savefig(path)

def output_bar_chart(xs, yss, path = 'dd.png'):
    plt.clf()
    # plt.bar(xs, ys, color ='maroon', width = 0.4)
    for ys in yss:
        plt.plot(xs, ys)
    plt.savefig(path)

# 0913
def draw_heatmap():
    mat, ys, xs = R3.get_mat()
    output_heatmap(mat, xs, ys)

# 0927
def draw_heatmap():
    xs, yss = R4.get_heatmap()
    output_bar_chart(xs, yss)

