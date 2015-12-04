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
      install_requires=['keras'],
      packages=find_packages())
