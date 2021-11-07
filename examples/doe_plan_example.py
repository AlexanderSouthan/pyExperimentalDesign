#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 00:29:14 2021

@author: almami
"""

from pyExperimentalDesign import doe_plan

doe = doe_plan(['Resp_1'],
               param_limits=[[50, 200], [1200, 1800], [2, 4], [9, 85]],
               param_levels=[['Yes', 'No']],
               param_cont=['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4'],
               param_categ=['Comp_5'],
               center_rep=6, factorial_rep=3, star_rep=3)

data = doe.simulate_response('Resp_1')