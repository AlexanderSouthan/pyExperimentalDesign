# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:37:27 2021

@author: Alexander Southan
"""

import numpy as np
import pandas as pd
import itertools


class doe_plan:
    def __init__(self, param_names, param_types, param_limits, response_names,
                 response_units=None, factorial_rep=3, center_rep=0):
        self.factorial_rep = factorial_rep
        self.center_rep = center_rep

        self.param_info = pd.DataFrame([], index=param_names)
        self.param_info['param_types'] = param_types
        param_limits = np.asarray(param_limits)
        self.param_info['param_lower'] = param_limits[:, 0]
        self.param_info['param_upper'] = param_limits[:, 1]

        self.data_table = pd.DataFrame([], columns=param_names + response_names)

        self.add_factorial_points()
        self.add_center_point()

    def add_factorial_points(self):
        factorial_df = pd.DataFrame(list(itertools.product(
            *self.param_info[['param_lower', 'param_upper']].values)),
            columns=self.param_info.index)

        self.data_table = pd.concat(
            [self.data_table] + [factorial_df]*self.factorial_rep,
            ignore_index=True)

    def add_center_point(self):
        center_df = self.param_info[['param_upper', 'param_lower']].mean(
            axis=1).to_frame().T

        self.data_table = pd.concat(
            [self.data_table] + [center_df]*self.center_rep,
            ignore_index=True)

    def add_star_points(self):
        # Here, alogic to select the star points needs to be implemented.
        pass

if __name__ == '__main__':

    doe = doe_plan(
        ['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4'],
        ['cont', 'cont', 'cont', 'cont'],
        [[50, 200], [1200, 1800], [2, 4], [9, 85]],
        ['Resp_1'], center_rep=3)
