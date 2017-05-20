# coding=utf-8
from enet import *


def select_model(model_name):
    if model_name == 'enet':
        import enet as model
    else:
        raise ValueError('Unknown model {}'.format(model_name))
    return model
