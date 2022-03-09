#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:51:57 2021

@author: Alexander Southan
"""

import unittest

from src.pyExperimentalDesign import doe_analysis, doe_plan


class TestExperimentalDesign(unittest.TestCase):

    def test_plots(self):
        doe_p = doe_plan(
            ['Resp_1'],
            param_limits=[[50, 200], [1200, 1800], [2, 4], [9, 85]],
            param_levels=[['Yes', 'No']],
            param_cont=['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4'],
            param_categ=['Comp_5'],
            center_rep=6, factorial_rep=3, star_rep=3)

        data, model_coefs = doe_p.simulate_response('Resp_1', noise_level=2)

        doe_anal = doe_analysis(
            data,
            ['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4', 'Comp_5'],
            ['cont', 'cont', 'cont', 'cont', 'categ'],
            ['Resp_1'], models=['2fi'], p_limits=[1])

        fig1, ax1 = doe_anal.actual_vs_predicted('Resp_1')
        fig2, ax2 = doe_anal.residual_vs_quantile('Resp_1')
        fig3, ax3 = doe_anal.residual_vs_run('Resp_1')
        fig4, ax4 = doe_anal.residual_vs_factor('Comp_1', 'Resp_1')
        fig5, ax5 = doe_anal.residual_vs_predicted('Resp_1')
        fig6, ax6 = doe_anal.box_cox_plot('Resp_1')
