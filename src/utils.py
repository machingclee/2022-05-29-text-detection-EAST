from pydash.objects import get, set_
import numpy as np
from shapely.geometry import Polygon


def iou(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = np.array([iou(S[i], S[t]) for t in order[1:]])

        inds = np.where(ious <= thres)[0]
        # since order[0] is taken out
        order = order[inds + 1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and iou(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


class ConsoleLog():
    def __init__(self, lines_up_on_end=0):
        self.CLR = "\x1B[0K"
        self.lines_up_on_batch_end = lines_up_on_end
        self.record = {}

    def UP(self, lines):
        return "\x1B[" + str(lines + 1) + "A"

    def DOWN(self, lines):
        return "\x1B[" + str(lines) + "B"

    def on_print_end(self):
        print(self.UP(self.lines_of_log))
        print(self.UP(self.lines_up_on_batch_end))

    def print(self, string_or_key_values, is_key_value=True):
        if not is_key_value:
            self.lines_of_log = 1

            print("".join(["\n"] * (self.lines_of_log)))
            print(self.UP(self.lines_of_log))

            print(string_or_key_values)

        else:
            lines_of_log = len(string_or_key_values)
            self.lines_of_log = lines_of_log

            # for the first time,
            # print self.lines_of_log number of lines to occupy the space
            print("".join(["\n"] * (self.lines_of_log)))
            print(self.UP(self.lines_of_log))

            for key, value in string_or_key_values:
                if key == "" and value == "":
                    print()
                else:
                    if key != "" and value != "":
                        prev_value = get(self.record, key, 0.)
                        curr_value = value
                        diff = curr_value - prev_value
                        sign = "+" if diff >= 0 else ""
                        print("{0: <35} {1: <30}".format(key, value) + sign + "{:.5f}".format(diff) + self.CLR)
                        set_(self.record, key, value)

        self.on_print_end()

    def clear_log_on_epoch_end(self):
        # usually before calling this line, print() has been run, therefore we are at the top of the log.
        for _ in range(self.lines_of_log):
            # clear lines
            print(self.CLR)
        # ready for next epoch
        print(self.UP(self.lines_of_log))

