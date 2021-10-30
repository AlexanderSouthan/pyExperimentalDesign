#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import pandas as pd
import unittest

from src.pyExperimentalDesign import experimental_design


class TestExperimentalDesign(unittest.TestCase):

    def test_experimental_design(self):

        param_1 = pd.Series([1, 1, 2, 2, 3, 3], name='param_1')
        param_2 = pd.Series([1, 1, 2, 2, 3, 3], name='param_2')
        response_1 = pd.Series([4, 5, 6, 2, 3, 4], name='response_1')
        data = pd.concat([param_1, param_2, response_1], axis=1)

        doe = experimental_design(
            data, ['param_1', 'param_2'], ['cont', 'cont'], ['response_1'])
