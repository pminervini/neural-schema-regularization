# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


setup(name='hyper',
      version='0.0.1',
      description='Energy-Based Latent Factor Models for Link Prediction in Knowledge Hypergraphs',
      author='Pasquale Minervini',
      author_email='p.minervini@gmail.com',
      url='https://github.com/pminervini/hyper',
      test_suite='tests',
      license='MIT',
      install_requires=[
          'keras>=1.0',
          'h5py>=2.5.0',
          'scikit-learn>=0.16.0',

          'requests>=2.8.1',
          'terminaltables>=2.1.0',
          'colorclass>=2.2.0',
          'matplotlib>=1.5.1',
          'ascii_graph>=1.1.4'
      ],
      packages=find_packages())
