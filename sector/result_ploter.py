import result_grid2 as R2
import result_090521 as R1
from plot import *

def draw():
    y2 = R2.get_e2_avgs(R2.rates, 16)
    y1 = R1.get_e2_avgs()
    draw_line_chart(AUXILIARY_LOSS_XS, [y1, y2], ['2nd', '1st'])
