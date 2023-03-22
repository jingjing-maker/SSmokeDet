#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name):
    import SSmokeNet

    ssmokenet_path = os.path.dirname(os.path.dirname(SSmokeNet.__file__))
    filedict = {
        "SSmokeNet":"ssmokenet.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(ssmokenet_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):

    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
