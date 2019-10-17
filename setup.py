# -*- coding: utf-8 -*-

#############################################
# Author: huangsir
# Mail: hxysir@163.com
# Created Time:  2019-10-17 15:00
#############################################

from setuptools import setup, find_packages

setup(
    name = "creditmodel",
    version = "1.0",
    keywords = ("pip", "EDA","Preprocessing", "FeatureEngineering", "Modeling","RiskStragety"),
    description = "build credit score model",
    long_description = "build credit score model",
    license = "MIT License",

    url = "https://github.com/huangxianyang/creditmodel",
    author = "huangsir",
    author_email = "hxysir@163.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ['pymongo==3.7.2','PyMySQL==0.9.3','missingno==0.4.1','pandas-profiling==1.4.1','numpy==1.15.0',
                        'matplotlib==3.0.2','pandas==0.23.4','scikit-learn==0.21.3','imbalanced-learn==0.5.0',
                        'statsmodels==0.9.0','scorecardpy==0.1.7','seaborn==0.9.0']
)