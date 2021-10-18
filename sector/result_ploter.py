# import result_grid2 as R2
# import result_090521 as R1
# import result_091321 as R3
# import result_092721 as R4
import result_merger_101821 as R5
from plot import *
import seaborn as sns # for heatmap
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# 0905
def draw():
    y2 = R2.get_e2_avgs(R2.rates, 16)
    y1 = R1.get_e2_avgs()
    draw_line_chart(AUXILIARY_LOSS_XS, [y1, y2], ['2nd', '1st'])

def output_heatmap(mat, xs, ys, path = 'dd.png'):
    plt.clf()
    sns.heatmap(mat, xticklabels=xs, yticklabels=ys, cmap='Blues', annot=True, fmt='.4g')
    plt.savefig(path)

def output_line_chart(xs, yss, legends, path = 'dd.png'):
    plt.clf()
    # plt.bar(xs, ys, color ='maroon', width = 0.4)
    for ys,legend in zip(yss, legends):
        plt.plot(xs, ys, label=legend)
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig(path)

# 0913
def draw_heatmap():
    mat, ys, xs = R3.get_mat()
    output_heatmap(mat, xs, ys)

# 0927
def draw_line_chart():
    xs, yss, legends = R4.get_heatmap()
    output_line_chart(xs, yss, legends)

# 1018 
def draw_line_chart():
    xs, yss, legends = R5.get_results()
    output_line_chart(xs, yss, legends)

