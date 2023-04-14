# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:37:27 2021

@author: Alexander Southan
"""

import numpy as np
import pandas as pd
import itertools

from pyDataFitting import model_tools


class doe_plan:
    def __init__(self, response_names, param_limits=[], param_levels=[], 
                 param_cont=None, param_categ=None, response_units=None,
                 factorial_rep=3, center_rep=3, star_rep=3, star_alpha=1):
        if (not param_limits) and (not param_levels):
            raise ValueError('No parameter limits or levels given. At least '
                             'one of the two must be provided.')
        if param_cont is None:
            param_cont = ['Param{}'.format(ii+1) for ii in
                          range(len(param_limits))]
        if param_categ is None:
            param_categ = ['Param{}'.format(jj+1) for jj in range(
                len(param_limits), len(param_limits)+len(param_levels))]

        param_cont_coded = ['P{}'.format(ii+1) for ii in
                            range(len(param_limits))]
        param_categ_coded = ['P{}'.format(jj+1) for jj in range(
            len(param_limits), len(param_limits)+len(param_levels))]


        self.reproductions = pd.Series([factorial_rep, center_rep, star_rep],
                                       index=['factorial', 'center', 'star'])
        self.star_alpha = star_alpha

        self.param_cont = pd.DataFrame(
            [], index=param_cont, columns=['lower', 'upper', 'coded_name'])
        self.param_categ = pd.DataFrame(
            [], index=param_categ, columns=['levels', 'coded_name',
                                            'coded_levels'])

        if param_limits:
            self.param_cont[['lower', 'upper']] = param_limits
            self.param_cont['coded_name'] = param_cont_coded
        if param_levels:
            self.param_categ['levels'] = param_levels
            self.param_categ['coded_name'] = param_categ_coded
            for curr_param in self.param_categ.index:
                self.param_categ.at[curr_param, 'coded_levels'] = np.linspace(
                    -1, 1, num=len(self.param_categ.at[curr_param, 'levels']),
                    endpoint=True)

        self.data_table = pd.DataFrame(
            [], columns=param_cont + param_categ + response_names +
            ['point_type'] + param_cont_coded + param_categ_coded)

        if param_limits:
            self.add_factorial_points()
            self.add_center_point()
            self.add_star_points()
        if param_levels:
            self.apply_categ_params()

    def add_factorial_points(self):
        factorial_df = pd.DataFrame(list(itertools.product(
            *self.param_cont[['lower', 'upper']].values)),
            columns=self.param_cont.index)
        factorial_df['point_type'] = 'factorial'

        coded_df = pd.DataFrame(list(itertools.product(
            *[[-1, 1] for _ in range(len(self.param_cont))])),
            columns=self.param_cont['coded_name'])

        factorial_df = pd.concat([factorial_df, coded_df], axis=1)

        self.data_table = pd.concat(
            [self.data_table] + [factorial_df]*self.reproductions['factorial'],
            ignore_index=True)

    def add_center_point(self):
        center_df = self.param_cont[['upper', 'lower']].mean(
            axis=1).to_frame().T
        center_df['point_type'] = 'center'

        coded_df = pd.DataFrame([[0]*len(self.param_cont)],
            columns=self.param_cont['coded_name'])

        center_df = pd.concat([center_df, coded_df], axis=1)

        self.data_table = pd.concat(
            [self.data_table] + [center_df]*self.reproductions['center'],
            ignore_index=True)

    def add_star_points(self):
        minus_alpha = np.repeat(
            [self.param_cont[['upper', 'lower']].mean(axis=1).values],
            len(self.param_cont), axis=0)
        np.fill_diagonal(
            minus_alpha, (self.param_cont['upper'].values -
                          self.param_cont['lower'].values)/2 *
            (-self.star_alpha-1)+self.param_cont['upper'].values)
        minus_alpha_df = pd.DataFrame(
            minus_alpha, columns=self.param_cont.index)
        minus_alpha_df['point_type'] = 'star'

        minus_coded = pd.DataFrame(
            np.diag([-self.star_alpha]*len(self.param_cont)),
            columns=self.param_cont['coded_name'])

        minus_alpha_df = pd.concat([minus_alpha_df, minus_coded], axis=1)

        plus_alpha = np.repeat(
            [self.param_cont[['upper', 'lower']].mean(axis=1).values],
            len(self.param_cont), axis=0)
        np.fill_diagonal(
            plus_alpha, (self.param_cont['upper'].values -
                         self.param_cont['lower'].values)/2 *
            (self.star_alpha-1)+self.param_cont['upper'].values)
        plus_alpha_df = pd.DataFrame(
            plus_alpha, columns=self.param_cont.index)
        plus_alpha_df['point_type'] = 'star'

        plus_coded = pd.DataFrame(
            np.diag([self.star_alpha]*len(self.param_cont)),
            columns=self.param_cont['coded_name'])

        plus_alpha_df = pd.concat([plus_alpha_df, plus_coded], axis=1)

        self.data_table = pd.concat(
            [self.data_table] +
            [minus_alpha_df, plus_alpha_df]*self.reproductions['star'],
            ignore_index=True)

    def apply_categ_params(self):
        for curr_param in self.param_categ.index:
            level_dfs = []
            curr_data_table = self.data_table.copy()
            if len(curr_data_table) == 0:
                curr_data_table.at[0, curr_param] = 'dummy_entry'
            for curr_level, curr_coded in zip(
                    self.param_categ.at[curr_param, 'levels'],
                    self.param_categ.at[curr_param, 'coded_levels']):
                curr_data_table[curr_param] = curr_level
                curr_data_table[
                    self.param_categ.at[curr_param, 'coded_name']] = curr_coded
                level_dfs.append(curr_data_table.copy())
            self.data_table = pd.concat(
                level_dfs, ignore_index=True)

    def simulate_response(self, response, noise_level=1):
        model = model_tools(
            '2fi',
            self.param_cont['coded_name'].to_list() +
            self.param_categ['coded_name'].to_list(),
            ['cont']*len(self.param_cont) + ['categ']*len(self.param_categ),
            response_name=response)

        model_coefs = pd.Series(np.random.rand(len(model.param_combinations)),
                                index=model.param_combinations.index)
        model_coefs['Intercept'] = np.random.rand()

        for curr_row in self.data_table.index:
            curr_coded = self.data_table.loc[
                curr_row, self.param_cont['coded_name'].to_list() +
                self.param_categ['coded_name'].to_list()].to_list()
            curr_front_factor = model.calc_front_factors(curr_coded)
            self.data_table.at[curr_row, response] = (
                curr_front_factor*model_coefs).sum()

        self.data_table[response] = pd.to_numeric(self.data_table[response])
        self.data_table[response] += noise_level*np.random.normal(
            size=len(self.data_table[response]))

        return (self.data_table, model_coefs)
