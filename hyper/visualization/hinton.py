#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import numpy as np
import termcolor
from colorclass import Color, Windows
from terminaltables import SingleTable, AsciiTable
from tabulate import tabulate

import logging


def hinton_diagram(arr, max_arr=None):
    max_arr = arr if max_arr is None else max_arr
    max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
    res = [list([hinton_diagram_value(x, max_val) for x in _arr]) for _arr in arr]
    return res


def hinton_diagram_value(val, max_val):
    chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    #chars = ['a', 'b', 'c']
    if abs(abs(val) - max_val) < 1e-8:
        step = len(chars) - 1
    else:
        step = int(abs(float(val) / max_val) * len(chars))
    attr = 'red' if val < 0 else 'green'
    return Color('{auto' + attr + '}' + str(chars[step]) + '{/auto' + attr + '}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data = np.random.randn(50, 100)
    print(data)
    diagram = hinton_diagram(data)
    table = SingleTable(diagram)
    table.inner_heading_row_border = False
    table.inner_footing_row_border = False
    table.inner_column_border = False
    table.inner_row_border = False
    table.column_max_width = 1
    print(table.table)
    #print(tabulate(diagram))
