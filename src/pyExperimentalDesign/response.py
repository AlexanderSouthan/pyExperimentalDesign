#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:21:05 2021

@author: Alexander Southan
"""

class response:
    def __init__(self, name, unit=None, **kwargs):
        self.name = name

        if unit is None:
            self.unit = 1
        else:
            self.unit = unit

        self.model_type = kwargs.get('model_type', None)
        self.model_tools = kwargs.get('model_tools', None)
        self.model = kwargs.get('model', None)
        self.anova_table = kwargs.get('anova_table', None)
        self.influences = kwargs.get('influences', None)
