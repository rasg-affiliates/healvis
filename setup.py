# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from setuptools import setup


def branch_scheme(version):
    """
    Local version scheme that adds the branch name for absolute reproducibility.
    If and when this is added to setuptools_scm this function and file can be removed.
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch == "main":
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            return version.format_choice("+{node}.{branch}", "+{node}.{branch}.dirty")


setup(use_scm_version={"local_scheme": branch_scheme})
