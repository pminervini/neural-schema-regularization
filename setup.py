from setuptools import setup
from setuptools import find_packages


setup(name='energy-hypergraph',
      version='0.0.1',
      description='Energy-Based Latent Factor Models for Link Prediction in Knowledge Hypergraphs',
      author='Pasquale Minervini',
      author_email='p.minervini@gmail.com',
      url='https://github.com/pminervini/energy-hypergraph',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages())
