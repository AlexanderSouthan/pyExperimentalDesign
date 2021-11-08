#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:51:34 2021

@author: almami
"""


import numpy as np


class parameter:
    def __init__(self, name, ptype='cont', unit=None, coded_name=None,
                 **kwargs):
        self.name = name
        self.type = ptype

        if unit is None:
            self.unit = 1
        else:
            self.unit = unit
        self.coded_name = coded_name

        if self.type == 'cont':
            limits = kwargs.get('limits')
            self.set_limits(limits)
            self.levels = None
        elif self.type == 'categ':
            self.set_limits(None)
            self.levels = kwargs.get('levels')

    def set_limits(self, limits=None):
        if limits is not None:
            self.limits = np.sort(np.asarray(limits))
            self.lower_bound = self.limits[0]
            self.upper_bound = self.limits[1]
        else:
            self.limits = None
            self.lower_bound = None
            self.upper_bound = None
