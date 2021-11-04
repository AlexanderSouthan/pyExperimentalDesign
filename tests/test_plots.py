#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:51:57 2021

@author: almami
"""

import pandas as pd
import unittest

from src.pyExperimentalDesign import experimental_design


class TestExperimentalDesign(unittest.TestCase):

    def test_plots(self):

        test_data = pd.read_csv('tests/doe_test_data.txt', sep='\t')

        doe = experimental_design(
            test_data, ['Comp_A', 'Comp_B', 'Comp_C'],
            ['cont', 'cont', 'cont'],
            ['Resp_1'], '2fi')

        fig1, ax1 = doe.actual_vs_predicted('Resp_1')
        fig2, ax2 = doe.residual_vs_quartile('Resp_1')
        fig3, ax3 = doe.residual_vs_run('Resp_1')
        fig4, ax4 = doe.residual_vs_factor('Comp_A', 'Resp_1')
        fig5, ax5 = doe.residual_vs_predicted('Resp_1')
        fig6, ax6 = doe.box_cox_plot('Resp_1')
