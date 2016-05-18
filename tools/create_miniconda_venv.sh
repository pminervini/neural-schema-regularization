#!/bin/bash

#
# This script is intended for use in continuous integration
# jobs to set up a new Python 3 virtual environment for
# testing and profiling, but it can be used in any situation
# to set up a virtual environment that will support kedi-link-prediction.
# A full Anaconda distrubtion will be installed.  Remaining
# dependencies will be met when pip installs kedi-link-prediction.
#
# Execute this script from the wp-tomoe-link_prediction directory.
# After this script runs it is still necessary to activate the
# new virtual environment in the calling script.
#
# usage:
#   create_miniconda_venv.sh <path to miniconda> <virtualenv name>
#

export MINICONDA3=$1
export VIRTUALENV=$2

# if the specified virtual environment exists then remove it
if [ -d ${MINICONDA3}/envs/${VIRTUALENV} ]; then
  ${MINICONDA3}/bin/conda remove -y --name ${VIRTUALENV} --all
fi

${MINICONDA3}/bin/conda create -qy --name ${VIRTUALENV} anaconda

source ${MINICONDA3}/bin/activate ${VIRTUALENV}

pip install ./evaluation-framework/kedi-link-prediction
