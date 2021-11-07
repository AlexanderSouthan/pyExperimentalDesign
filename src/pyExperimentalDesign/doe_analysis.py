# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 00:10:14 2021.

@author: Alexander Southan
"""

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt

from .model_tools import model_tools


class doe_analysis:
    def __init__(self, data, param_columns, param_types, response_columns,
                 response_units=None, models=None, p_limit=1):
        """
        Initialize an experimental_design instance.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing the data for the parameter values and the
            measured responses in the columns.
        param_columns : list
            A list containing the column labels of data that contain the
            parameter values.
        param_types : list of string
            A list defining the parameter type of the different parameters.
            Allowed values are 'cont' for continuous/numerical parameters and
            'categ' for categorial parameters, i.e. parameters for which only
            certain values are allowed. Must have the same number of elements
            like param_columns.
        response_columns : list
            A list containing the column labels of data that contain the
            response values.
        response_units : list, optional
            A list containing the units of the responses. If provided, must
            have the same number of elements like response_columns. The default
            is None, meaning no unit for all responses (1).
        models : list of string, optional
            A list defining the models used for initial analysis. If provided,
            must have the same number of elements like response_columns. The
            default is None, meaning a two-factor interaction ('2fi') model for
            all responses. Other allowed values are in self.model_types.
        p_limit : float, optional
            The upper threshold of allowed p values of the model terms. All
            model terms with p values higher than the threshold will be
            excluded from the analysis. Default is 1, resuling in no exclusion
            of any model term.

        Returns
        -------
        None.

        """
        if not (len(param_columns) == len(param_types)):
            raise ValueError('param_columns and param_types should have equal '
                             'length, but are {} and {}, respectively.'.format(
                                 len(param_columns), len(param_types)))

        self.data = data
        # The next step is important for input DataFrames that have a column
        # Dtype of object which might occur in mixed Dtype DataFrames.
        for curr_col in response_columns:
            self.data[curr_col] = pd.to_numeric(self.data[curr_col])
        self.p_limit = p_limit

        self.param_info = pd.DataFrame([], index=param_columns)
        self.param_info['coded_name'] = ['P{}'.format(ii+1)
                                         for ii in range(len(param_columns))]
        self.param_info['lower_bound'] = data[param_columns].min()
        self.param_info['upper_bound'] = data[param_columns].max()
        self.param_info['type'] = param_types

        # Careful when adding another model, all references to the following
        # list have to be updated
        self.model_types = ['linear', '2fi', '3fi', 'quadratic']

        self.response_info = pd.DataFrame([], index=response_columns)
        self.response_info['coded_name'] = [
            'R{}'.format(ii+1) for ii in range(len(response_columns))]
        if response_units is None:
            self.response_info['unit'] = 1
        else:
            self.response_info['unit'] = response_units
        self.response_info['model_type'] = None
        self.response_info['model'] = None
        self.response_info['model_tools'] = None
        if models is None:
            models = ['2fi']*len(response_columns)
        self.replace_model(response_columns, models)

        self.encode_cont_columns()
        self.encode_categ_columns(['Yes', 'No'], [-1, 1])

        self.perform_anova()

    def data_mean(self):
        """
        Calculate the mean of sample reproductions.

        Returns
        -------
        DataFrame
            A DataFrame containing the means of sample reproductions for all
            responses. The index is a MultiIndex with levels of the coded
            parameter values.

        """
        return self.data.groupby(
            self.param_info['coded_name'].to_list()).mean()

    def data_std(self):
        """
        Calculate the standard deviation of sample reproductions.

        Returns
        -------
        DataFrame
            A DataFrame containing the standard deviations of sample
            reproductions for all responses (columns). The index is a
            MultiIndex with levels of the coded parameter values.

        """
        return self.data.groupby(
            self.param_info['coded_name'].to_list()).std()

    def data_n(self):
        """
        Calculate the number of reproductions for all parameter settings.

        Returns
        -------
        Series
            A DataFrame containing the number of reproductions for each set of
            parameter settings. The index is a MultiIndex with levels of the
            coded parameter values.

        """
        return self.data.groupby(
            self.param_info['coded_name'].to_list()).size()

    def replace_model(self, responses, new_models):
        """
        Apply a certain model to a set of responses.

        Parameters
        ----------
        responses : list
            A list of responses for which the new_models are used. Must be
            elements of self.response_info.index.
        new_models : list of string
            A list of the new models used for data analysis. Must have the same
            number of elements like responses and only values in
            self.model_types are allowed.

        Returns
        -------
        None.

        """
        if len(responses) != len(new_models):
            raise ValueError('responses and new_models should have equal '
                             'length, but are {} and {}, respectively.'.format(
                                 len(responses), len(new_models)))
        if not set(responses).issubset(self.response_info.index.to_list()):
            raise ValueError(
                'At least one element of responses is not valid. {} was '
                'given, and only elements from {} are allowed.'.format(
                    responses, self.response_info.index.to_list()))
        if not set(new_models).issubset(self.model_types):
            raise ValueError(
                'At least one element of new_models is not valid. {} was '
                'given, and only elements from {} are allowed.'.format(
                    new_models, self.model_types))

        model_tools_all = []
        for curr_response, curr_model in zip(responses, new_models):
            model_tools_all.append(
                model_tools(curr_model, self.param_info['coded_name'].values,
                            param_types=self.param_info['type'].values,
                            response_name=curr_response))

        self.response_info.loc[responses, 'model_type'] = new_models
        self.response_info.loc[responses, 'model_tools'] = (
            model_tools_all)

    def encode_cont_columns(self):
        """
        Encode the continuous/numerical parameters to a scale between -1 and 1.

        Returns
        -------
        None.

        """
        for curr_row in self.param_info[
                self.param_info['type'] == 'cont'].index.values:
            max_value = self.param_info.at[curr_row, 'upper_bound']
            min_value = self.param_info.at[curr_row, 'lower_bound']
            self.data[self.param_info.at[curr_row, 'coded_name']] = (
                2/(max_value-min_value)*(self.data[curr_row]-max_value)+1
                ).astype('float').round(10)

    def encode_categ_columns(self, levels, coded_values):
        """
        Encode categorial parameters.

        Parameters
        ----------
        levels : list
            The allowed levels of the categorial parameters.
        coded_values : list of numbers
            The numerical values used to encode the different values of the
            categorial parameters.

        Returns
        -------
        None.

        """
        for curr_col in self.param_info[
                self.param_info['type'] == 'categ'].index.values:
            self.data[self.param_info.at[curr_col, 'coded_name']] = np.nan
            for curr_level, curr_coded in zip(levels, coded_values):
                self.data.loc[
                    self.data[curr_col] == curr_level,
                    self.param_info.at[curr_col, 'coded_name']] = curr_coded

    def perform_anova(self, model=None):
        """
        Perform the actual ANOVA.

        The ANOVA is done by the statsmodels anova_lm method. For this purpose,
        a model is used that is not necessarily linear.

        Parameters
        ----------
        model : string, optional
            If a model is given, the model is used for the calculations.
            Otherwise, the model passed upon initialization is used. The
            default is None.

        Returns
        -------
        None.

        """
        if model is not None:
            if model in self.model_types:
                self.replace_model(model)
            else:
                raise ValueError('No valid model given. \'{}\' was given, '
                                 'but it should be an element of {}.'.format(
                                     model, self.model_types))

        self.response_info['model'] = None
        self.response_info['anova_tables'] = None
        self.response_info['influences'] = None
        for curr_response in self.response_info.index.values:
            counter = 1
            combi_mask = pd.Series(True, index=self.response_info.at[
                curr_response, 'model_tools'].param_combinations.index)
            while True:
                # Generate model
                # Categorial variables can be added with C(variable)
                # see https://www.statsmodels.org/dev/example_formulas.html
                # An intercept is added by default
                curr_model_tools = self.response_info.at[
                    curr_response, 'model_tools']
                curr_model = ols(
                    curr_model_tools.model_string(combi_mask=combi_mask),
                    data=self.data).fit()
                # Perform the ANOVA
                curr_anova = (sm.stats.anova_lm(curr_model, typ=2))

                par_c = curr_model_tools.param_combinations
                curr_p_mask = ~curr_anova.index.isin(
                    ['Residual'] + par_c[par_c[
                        'for_hierarchy'] == True].index.to_list())
                curr_p = curr_anova.loc[curr_p_mask, 'PR(>F)']
                if (curr_p <= self.p_limit).all():
                    print('ANOVA runs for {}: {}'.format(curr_response,
                                                         counter))
                    break
                else:
                    combi_mask = curr_p <= self.p_limit
                    combi_mask = combi_mask.reindex(self.response_info.at[
                        curr_response,
                        'model_tools'].param_combinations.index,
                        fill_value=False)
                    counter += 1

            self.response_info.at[curr_response, 'model'] = curr_model
            self.response_info.at[curr_response, 'anova_tables'] = curr_anova
            self.response_info.at[curr_response, 'influences'] = (
                self.response_info.at[curr_response,
                                      'model'].get_influence())
            self.data[curr_response + '_fitted'] = self.response_info.at[
                curr_response, 'model'].fittedvalues

    def calc_model_value(self, response, coded_values):
        """
        Calculate one value the regression model predicts.

        Calculation is done for one set of parameter settings.

        Parameters
        ----------
        response : string
            The name of the response for which the model values are calculated.
            Must be an element of self.response_info.index.
        coded_values : list of float
            A list containing one set of coded parameter values. Must contain
            as many elements as there are parameters in the model.

        Returns
        -------
        float
            The predicted response value for the parameter settings.

        """
        front_factors = self.response_info.at[
            response, 'model_tools'].calc_front_factors(coded_values)
        return (front_factors *
                self.model_coefs(response)).sum()

    def model_coefs(self, response):
        return self.response_info.at[response, 'model'].params

    def model_metrics(self):
        """
        Return some relevant metrics about the model used.

        Returns
        -------
        model_metrics : DataFrame
            A DataFrame containing the R^2, R^2_adj, RMSE and model equation
            for all responses.

        """
        model_metrics = pd.DataFrame([], columns=['R^2', 'R^2_adj', 'RMSE',
                                                  'model_equation'])
        for curr_response in self.response_info.index:
            model_metrics.at[curr_response, 'R^2'] = self.response_info.at[
                curr_response, 'model'].rsquared
            model_metrics.at[curr_response, 'R^2_adj'] = self.response_info.at[
                curr_response, 'model'].rsquared_adj
            model_metrics.at[curr_response, 'RMSE'] = rmse(
                self.predicted()[curr_response], self.actual()[curr_response])
            model_metrics.at[curr_response, 'model_equation'] = (
                self.response_info.at[
                    curr_response, 'model_tools'].model_string())
        return model_metrics

    def anova_table(self, response):
        """
        Return the ANOVA table for one response.

        Parameters
        ----------
        response : string
            The name of the response for which the ANOVA table is returned.

        Returns
        -------
        DataFrame
            The ANOVA table.

        """
        return self.response_info.at[response, 'anova_tables']

    def residuals(self, kind='externally_studentized'):
        """
        Return the the residuals of the model used for data analysis.

        Parameters
        ----------
        kind : string, optional
            The kind of the residuals used for the calculations. Possible
            values are 'externally_studentized', 'internally_studentized', and
            'plain'. The default is 'externally_studentized'.

        Returns
        -------
        resids : DataFrame
            A DataFrame containing the residuals of all responses (columns).
            Has the same index like self.data.

        """
        resids = pd.DataFrame([], columns=self.response_info.index,
                              index=self.data.index, dtype='float')
        for curr_response in resids.columns:
            if kind == 'externally_studentized':
                resids[curr_response] = self.response_info.at[
                    curr_response, 'influences'].resid_studentized_external
            elif kind == 'internally_studentized':
                resids[curr_response] = self.response_info.at[
                    curr_response, 'influences'].resid_studentized_internal
            elif kind == 'plain':
                resids[curr_response] = self.response_info.at[
                    curr_response, 'influences'].resid
            else:
                raise ValueError('Value for kind is \'{}\', but must be one of'
                                 ' the following strings: '
                                 '\'externally_studentized\', '
                                 '\'internally_studentized\', or '
                                 '\'plain\'.'.format(kind))
        return resids

    def residual_percentiles(self, kind='externally_studentized'):
        """
        Calculate the percentiles of the residuals.

        Parameters
        ----------
        kind : string, optional
            The kind of the residuals used for the calculations. Possible
            values are 'externally_studentized', 'internally_studentized', and
            'plain'. The default is 'externally_studentized'.

        Returns
        -------
        resid_percentiles : DataFrame
            A DataFrame containing the percentiles of the residuals of all
            responses (columns). Has the same index like self.data.

        """
        resids = self.residuals(kind)
        resid_percentiles = pd.DataFrame([], columns=resids.columns,
                                         index=self.data.index, dtype='float')
        for curr_response in resid_percentiles.columns:
            for curr_idx in resids.index:
                resid_percentiles.at[curr_idx, curr_response] = (
                    scipy.stats.percentileofscore(
                        resids[curr_response],
                        resids.at[curr_idx, curr_response], kind='mean'))
        return resid_percentiles

    def theo_residual_percentiles(self, kind='externally_studentized'):
        """
        Calculate the theoretical percentiles of the residuals.

        Parameters
        ----------
        kind : string, optional
            The kind of the residuals used for the calculations. Possible
            values are 'externally_studentized', 'internally_studentized', and
            'plain'. The default is 'externally_studentized'.

        Returns
        -------
        theo_resid_percentiles : DataFrame
            A DataFrame containing the theoretical percentiles of the
            residuals of all responses (columns). Has the same index like
            self.data.

        """
        resid_percentiles = self.residual_percentiles(kind)
        theo_resid_percentiles = pd.DataFrame(
            [], columns=resid_percentiles.columns, index=self.data.index,
            dtype='float')
        for curr_response in theo_resid_percentiles.columns:
            theo_resid_percentiles[curr_response] = (
                np.sqrt(2)*scipy.special.erfinv(
                    2*resid_percentiles[curr_response].values/100-1))
        return theo_resid_percentiles

    def leverages(self):
        leverages = pd.DataFrame([], columns=self.response_info.index,
                                 index=self.data.index, dtype='float')
        for curr_response in leverages.columns:
            leverages[curr_response] = self.response_info.at[
                curr_response, 'influences'].hat_matrix_diag
        return leverages

    def cooks_distances(self):
        cooks_distances = pd.DataFrame([], columns=self.response_info.index,
                                       index=self.data.index, dtype='float')
        for curr_response in cooks_distances.columns:
            cooks_distances[curr_response] = self.response_info.at[
                curr_response, 'influences'].cooks_distance[0]
        return cooks_distances

    def dffits(self):
        dffits = pd.DataFrame([], columns=self.response_info.index,
                              index=self.data.index, dtype='float')
        for curr_response in dffits.columns:
            dffits[curr_response] = self.response_info.at[
                curr_response, 'influences'].dffits[0]
        return dffits

    def dfbetas(self):
        dfbetas = pd.DataFrame([], columns=self.response_info.index,
                               index=self.data.index, dtype='float')
        for curr_response in dfbetas.columns:
            dfbetas[curr_response] = self.response_info.at[
                curr_response, 'influences'].dfbetas
        return dfbetas

    def actual(self):
        return self.data[self.response_info.index]

    def predicted(self):
        predicted = self.data[self.response_info.index + '_fitted']
        predicted.columns = self.response_info.index
        return predicted

    def actual_vs_predicted(self, response):
        fig, ax = plt.subplots()
        ax.plot(self.predicted()[response], self.actual()[response], ls='none',
                marker='o')
        x_min = min([self.predicted()[response].min(),
                     self.actual()[response].min()])
        x_max = max([self.predicted()[response].max(),
                     self.actual()[response].max()])
        ax.plot(np.linspace(x_min, x_max), np.linspace(x_min, x_max))
        ax.set_xlabel(response + ' (predicted)')
        ax.set_ylabel(response + ' (actual)')
        ax.set_title('Actual vs. Predicted')

        return (fig, ax)

    def residual_vs_quantile(self, response,
                             residual_kind='externally_studentized'):
        fig, ax = plt.subplots()
        resid = self.residuals(kind=residual_kind)[response]
        theo_percentile = self.theo_residual_percentiles(
            kind=residual_kind)[response]
        ax.plot(resid, theo_percentile, ls='none', marker='o')
        x_min = min([resid.min(), theo_percentile.min()])
        x_max = max([resid.max(), theo_percentile.max()])
        ax.plot(np.linspace(x_min, x_max), np.linspace(x_min, x_max))
        ax.set_xlabel(response + ' ' + residual_kind + ' residuals')
        ax.set_ylabel('Theoretical quantiles')
        ax.set_title('Normal plot of residuals')

        return (fig, ax)

    def residual_vs_run(self, response,
                        residual_kind='externally_studentized'):
        fig, ax = plt.subplots()
        ax.plot(self.residuals(kind=residual_kind)[response],
                ls='-', marker='o')
        ax.set_xlabel('Run')
        ax.set_ylabel(response + ' ' + residual_kind + ' residuals')
        ax.set_title('Residuals vs. Run')
        ax.axhline(0, c='k', ls='--')
        ax.axhline(3, c='r', ls='--')
        ax.axhline(-3, c='r', ls='--')

        return (fig, ax)

    def residual_vs_factor(self, param, response,
                           residual_kind='externally_studentized'):
        fig, ax = plt.subplots()
        ax.plot(self.data[param], self.residuals(
            kind=residual_kind)[response], ls='none', marker='o')
        ax.set_xlabel(param)
        ax.set_ylabel(response + ' ' + residual_kind + ' residuals')
        ax.set_title('Residuals vs. parameter')
        ax.axhline(0, c='k', ls='--')
        ax.axhline(3, c='r', ls='--')
        ax.axhline(-3, c='r', ls='--')

        return (fig, ax)

    def residual_vs_predicted(self, response,
                              residual_kind='externally_studentized'):
        fig, ax = plt.subplots()
        ax.plot(self.predicted()[response],
                self.residuals(kind=residual_kind)[response], ls='none',
                marker='o')
        ax.set_xlabel(response + ' (predicted)')
        ax.set_ylabel(response + ' ' + residual_kind + ' residuals')
        ax.set_title('Residuals vs. predicted')
        ax.axhline(0, c='k', ls='--')
        ax.axhline(3, c='r', ls='--')
        ax.axhline(-3, c='r', ls='--')

        return (fig, ax)

    def box_cox_plot(self, response):
        fig, ax = plt.subplots()
        lambdas = np.linspace(-3, 3, 50)
        bc_llf = np.empty_like(lambdas)
        for ii, curr_lambda in enumerate(lambdas):
            curr_data = self.data[response].copy()
            if (curr_data < 0).any():
                # Make sure that data is positive
                curr_data -= curr_data.min()-1
            bc_llf[ii] = scipy.stats.boxcox_llf(curr_lambda,
                                                curr_data)
        _, best_lambda, conf_int = scipy.stats.boxcox(curr_data,
                                                      alpha=0.05)
        ax.axvline(
            best_lambda, c='k', ls='--', label='$\lambda=${}$\pm${}'.format(
                round(best_lambda, 3), round(best_lambda-conf_int[0], 3)))
        ax.axvline(conf_int[0], c='r', ls='--')
        ax.axvline(conf_int[1], c='r', ls='--')
        ax.plot(lambdas, bc_llf)
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('Box-Cox log-likelihood function')
        ax.set_title('Box-Cox plot for power transformation')
        ax.legend()

        return (fig, ax)
