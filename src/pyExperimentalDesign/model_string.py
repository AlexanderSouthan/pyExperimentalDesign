# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:26:39 2021

@author: Alexander Southan
"""

import numpy as np
import pandas as pd
import itertools


class model_string:
    def __init__(self, model_type, param_names, param_types=None,
                 response_name='R'):
        """
        Initialize model_string instance.

        Parameters
        ----------
        model_type : string
            The model type, must be an element of self.model_types, so
            currently ['linear', '2fi', '3fi', 'quadratic'].
        param_names : list of string
            The parameter names used in the model string.
        param_types : None or list of string, optional
            Defines the type of the parameters. Can be a list with as many
            entries as param_names. Allowed entries are 'cont' for continuous
            parameters and 'categ' for categorial parameters. The default is
            None, meaning that all parameters are continuous.
        response_name : string, optional
            The name of the response. The default is 'R'.

        Raises
        ------
        ValueError
            If invalid model_Type is given.

        Returns
        -------
        None.

        """
        # Careful when adding another model, all references to the following
        # list have to be updated
        self.model_types = ['linear', '2fi', '3fi', 'quadratic']
        self.model_type = model_type
        self.param_names = param_names
        if param_types is None:
            self.param_types = ['cont']*len(param_names)
        else:
            self.param_types = param_types
        self.response_name = response_name

        if self.model_type not in self.model_types:
            raise ValueError('No valid model_type given. Should be an element '
                             'of {}, but is \'{}\'.'.format(
                                 self.model_types, self.model_type))

        param_numbers = np.arange(len(self.param_names))

        self.param_combinations = pd.DataFrame(
            [], index=self.param_names, columns=np.append(
                self.param_names, ['string', 'mask', 'for_hierarchy']))

        # linear part is used for all models
        # for curr_name in self.param_names:
        combi_array = np.diag(np.ones_like(self.param_names))
        self.param_combinations.loc[self.param_names, 'string'] = (
            self.param_names)

        # two-factor interactions
        if self.model_type in self.model_types[1:4]: #  '2fi', '3fi', 'quadratic'
            for subset in itertools.combinations(param_numbers, 2):
                curr_idx = ':'.join(i for i in self.param_names[list(subset)])
                curr_string = '*'.join(i for i in self.param_names[list(subset)])
                self.param_combinations.at[curr_idx, 'string'] = curr_string

                curr_combi = np.zeros_like(self.param_names)
                curr_combi[list(subset)] = 1
                combi_array = np.append(combi_array, [curr_combi], axis=0)

        # three-factor interactions
        if self.model_type == self.model_types[2]:  # '3fi'
            for subset in itertools.combinations(param_numbers, 3):
                curr_idx = ':'.join(i for i in self.param_names[list(subset)])
                curr_string = '*'.join(i for i in self.param_names[list(subset)])
                self.param_combinations.at[curr_idx, 'string'] = curr_string

                curr_combi = np.zeros_like(self.param_names)
                curr_combi[list(subset)] = 1
                combi_array = np.append(combi_array, [curr_combi], axis=0)

        # quadratic terms
        if self.model_type == self.model_types[3]:  # 'quadratic'
            for curr_name in self.param_names:
                curr_idx = 'I({} * {})'.format(curr_name, curr_name)
                curr_string = 'I({}*{})'.format(curr_name, curr_name)
                self.param_combinations.at[curr_idx, 'string'] = curr_string

            curr_combi = np.diag(np.full_like(self.param_names, 2))
            combi_array = np.append(combi_array, curr_combi, axis=0)

        self.param_combinations[self.param_names] = combi_array
        self.param_combinations['mask'] = True
        self.param_combinations['for_hierarchy'] = False

        # Drop quadratic terms for categoric factors
        if self.model_type == self.model_types[3]:  # 'quadratic'
            for curr_name, curr_type in zip(self.param_names,
                                            self.param_types):
                if curr_type == 'categ':
                    self.param_combinations.drop('I({} * {})'.format(
                        curr_name, curr_name), inplace=True, axis=0)

    def model_string(self, combi_mask=None, check_hierarchy=True):
        """
        Generate the model string necessary for OLS fitting.

        Parameters
        ----------
        combi_mask : pd.Series
            A series containing boolean values which define if certain
            parameter combinations are included into the model string. The
            index should be identical to the index of self.param_combinations.
            The default is None, meaning that all parameter combinations are
            included into the model string.
        check_hierarchy : bool, optional
            Defines if the model hierarchy is checked and corrected if
            necessary. The deafult is True.

        Returns
        -------
        model_string : string
            The model string in the correct format to be used by the OLS
            function in experimental_design.perform_anova.

        """
        if combi_mask is not None:
            self.param_combinations['mask'] = combi_mask
        if check_hierarchy:
            self.check_hierarchy()

        return '{} ~ {}'.format(
            self.response_name, self.param_combinations.loc[
                self.param_combinations['mask'], 'string'].str.cat(sep=' + '))

    def check_hierarchy(self):
        """
        Check for hierarchy of the model implemented.
        
        All entries with False in self.param_combinations['mask'] are checked
        if they should be included in the model for hierarchy. If this is
        found for a parameter or a parameter combination, the corresponding
        entry in the DataFrame is set to True and the value in the column
        'for_hierarchy' is also set to True in order to show that this term is
        only included for hierarchy and not due to a significant contribution.

        Returns
        -------
        None.

        """
        self.param_combinations['for_hierarchy'] = False
        excluded_mask = ~self.param_combinations['mask']
        combi_data = self.param_combinations[self.param_names]
        check_data = self.param_combinations.loc[excluded_mask,
                                                 self.param_names]

        for curr_combi in check_data.index:
            curr_mask = check_data.loc[curr_combi]>0
            curr_hier = combi_data.where(combi_data.loc[:, curr_mask] > 0, 0)
            deps = curr_hier.merge(
                check_data.loc[[curr_combi]], indicator=True, how='left',
                on=curr_hier.columns.to_list())['_merge']=='both'
            deps.index = curr_hier.index
            if (deps*self.param_combinations['mask']).any():
                self.param_combinations.at[curr_combi, 'for_hierarchy'] = True
                self.param_combinations.at[curr_combi, 'mask'] = True
