#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:51:57 2021

@author: almami
"""

import pandas as pd

from pyExperimentalDesign import experimental_design

test_data = pd.read_csv('../tests/doe_test_data.txt', sep='\t')

doe = experimental_design(
    test_data, ['Comp_A', 'Comp_B', 'Comp_C'], ['cont', 'cont', 'cont'],
    ['Resp_1'], '2fi')


