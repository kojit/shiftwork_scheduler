#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set ts=4 sts=4 sw=4 expandtab fenc=utf-8 ff=unix :
#
# Schedule work shift
#
# Copyright (C) 2018 Koji Tashiro

import logging
import argparse
import numpy as np
import pandas as pd
import pulp
from pulp import lpSum, lpDot, LpBinary, LpVariable
from pulp import LpProblem, LpMaximize, LpStatus

logger = logging.getLogger(__name__)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option('display.unicode.east_asian_width', True)


def new_binary(count=[0]):
    count[0] += 1
    return LpVariable('b%.6d' % count[0], cat=LpBinary, lowBound=0)


def get_constraint_days(df, n_types):
    exact_days = dict()
    min_days = dict()
    max_days = dict()
    n_rows_cut = 1
    for i in range(1, df.shape[0]):
        name = df.iloc[i].name
        if name.startswith('num_'):
            exact_days[name[-1]] = list(df.iloc[i, (n_types * 2):].astype(int))
        elif name.startswith('min_'):
            min_days[name[-1]] = list(df.iloc[i, (n_types * 2):].astype(int))
        elif name.startswith('max_'):
            max_days[name[-1]] = list(df.iloc[i, (n_types * 2):].astype(int))
        else:
            break
        n_rows_cut += 1
    return n_rows_cut, exact_days, min_days, max_days


def schedule_shift(df, shift_types, good_patterns, prohibit_patterns, max_intervals):
    """Schedule Shift work and returns scheduled pandas DataFrame

    df: pandas DataFrame
    shift_types: a string of shift type names, shift type name must be a
                    one character. e.g. 'DNGO'(Day, Night,
                    Graveyard, Off, etc), '日夜明休'
    good_patterns: a list of good patterns(shift type character sequence)
    prohibit_patterns: a list of prohibited patterns
    max_intervals: a dict of max num of interval days in each shift
    """
    df = df.fillna(0).T     # NaN -> 0 and transpose
    #logger.debug('\n{}'.format(df))

    n_types = len(shift_types)
    leaders = [1 if l != 0 else 0 for l in df.iloc[0, (n_types * 2):]]
    n_rows_cut, exact_days, min_days, max_days = get_constraint_days(df, n_types)

    min_workers = dict()
    max_workers = dict()
    for i, c in enumerate(shift_types):
        min_workers[c] = list(df.iloc[n_rows_cut:, i].astype(int))
        max_workers[c] = list(df.iloc[n_rows_cut:, i + n_types].astype(int))

    df = df.iloc[n_rows_cut:, (n_types * 2):]
    logger.debug('\n{}'.format(df))

    # shift name -> shift binary list
    def shift_binary_list(s):
        l = [0] * n_types
        i = shift_types.find(str(s))
        if i >= 0: l[i] = 1
        return l
    df = df.applymap(shift_binary_list)

    n_days = df.shape[0]
    n_workers = df.shape[1]
    logger.debug('num of days:{}, workers:{}'.format(n_days, n_workers))

    # Variables
    v_assignment = np.array([[[new_binary() for _ in range(n_types)] for _ in range(n_workers)] for _ in range(n_days)])
    # supplement
    v_good_patterns = [[new_binary() for _ in range(n_workers)] for _ in range(n_days)]

    m = LpProblem(sense=LpMaximize)     # Solver

    indexes = list(df.index)
    match_request = lambda row: lpDot(row, v_assignment[indexes.index(row.name)])

    # Objective function
    m += (lpSum(df.apply(match_request, axis=1)) - lpSum(v_good_patterns))

    # Add constraints
    for i in range(n_days):
        for j in range(n_workers):
            m += lpSum(v_assignment[i][j]) == 1     # must assign something
        for k, v in min_workers.items():
            m += lpSum(v_assignment[i][j][shift_types.find(k)] for j in range(n_workers)) >= v[i]
        for k, v in max_workers.items():
            m += lpSum(v_assignment[i][j][shift_types.find(k)] for j in range(n_workers)) <= v[i]
        # at least one leader needed for each shift
        for k in range(n_types):
            m += lpDot([v_assignment[i][j][k] for j in range(n_workers)], leaders) >= 1

    # The 1st row can't be changed, because it's already scheduled.
    m += lpDot(v_assignment[0], df.iloc[0]) == n_workers

    for i in range(n_workers):
        for k, v in exact_days.items():
            m += lpSum(v_assignment[j][i][shift_types.find(k)] for j in range(n_days)) == v[i]
        for k, v in min_days.items():
            m += lpSum(v_assignment[j][i][shift_types.find(k)] for j in range(n_days)) >= v[i]
        for k, v in max_days.items():
            m += lpSum(v_assignment[j][i][shift_types.find(k)] for j in range(n_days)) <= v[i]
        for k, v in max_intervals.items():
            for j in range(n_days - v + 1):
                m += lpSum(v_assignment[j + l][i][shift_types.find(k)] for l in range(v)) >= 1

    for pattern in prohibit_patterns:
        n = len(pattern)
        index_pattern = [shift_types.find(s) for s in pattern]
        for j in range(n_workers):
            for i in range(n_days - n + 1):
                m += lpSum(v_assignment[i + h][j][index_pattern[h]] for h in range(n)) <= n - 1

    for pattern in good_patterns:
        n = len(pattern)
        if n > 2:
            raise ValueError('good pattern must not exceed 2 characters.')
        index_pattern = [shift_binary_list(s) for s in pattern]
        for j in range(n_workers):
            for i in range(n_days - 1):
                m += lpDot(index_pattern[0], v_assignment[i, j]) <= lpDot(index_pattern[1], v_assignment[i+1, j]) + v_good_patterns[i+1][j]

    status = m.solve()
    if status != 1:
        logger.error('There is no combination you want. Decrease offs or '
            'required number of people, or increase number of workers.')
        return

    logger.debug('Status {}'.format(LpStatus[status]))
    logger.debug('Objective function {}'.format(pulp.value(m.objective)))
    results = [[int(pulp.value(lpDot(range(n_types), v_assignment[i][j]))) for j in range(n_workers)] for i in range(n_days)]
    data = np.vectorize(lambda i: shift_types[i])(results)
    df = pd.DataFrame(data, columns=df.columns, index=df.index)

    for s in shift_types:
        df['%s' % s] = (df.iloc[:, :n_workers] == s).sum(1)
    df = df.T
    for s in shift_types:
        df['%s' % s] = (df.iloc[:, :n_days] == s).sum(1)
    logger.debug('\n{}'.format(df))
    return df


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--request', default='shift_request_en.txt',
            help='path to a tab separated shift request file. ')
    parser.add_argument('-i', '--iexcel', type=str, default=None, help='Input Excel filename.')
    parser.add_argument('-o', '--oexcel', type=str, default=None, help='Output Excel filename.')
    args = parser.parse_args()

    if args.iexcel:
        df = pd.read_excel(args.iexcel, index_col=0)
    else:
        df = pd.read_csv(args.request, delimiter='\t', index_col=0)

    shift_types = 'ODNG'    # Off, Day, Night, Graveyard
    good_patterns = ['GO']
    prohibit_patterns = ['ND', 'NO', 'NN', 'OG', 'DG', 'GG']
    max_intervals = {'O': 6}

    """
    shift_types = '休日夜明'    # 'ODNG'
    good_patterns = ['明休']
    prohibit_patterns = ['夜日', '夜休', '夜夜', '休明', '日明', '明明']
    max_intervals = {'休': 6}
    """

    df = schedule_shift(df, shift_types, good_patterns, prohibit_patterns, max_intervals)
    if args.oexcel:
        df.to_excel(args.oexcel)
