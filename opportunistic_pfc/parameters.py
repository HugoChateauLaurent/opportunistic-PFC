import numpy as np
import pickle
from itertools import chain, combinations, product

from . import model

class Parameter(object):

    def __init__(self, name, default_value, comment=None):
        super(Parameter, self).__init__()
        self.name = name
        self.default_value = default_value
        self.exploration_values = []
        self.comment = comment

    @property
    def default(self):
        return len(self.exploration_values) == 0
