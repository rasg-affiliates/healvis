set -xe

# want 1 script to rule them all
# but this part is not needed on MACOS
sudo apt-get update
sudo apt-get install -y gcc g++
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --name=${ENV_NAME}  python=$PYTHON --quiet
conda env update -f ci/${ENV_NAME}.yaml
source activate ${ENV_NAME}
conda list -n ${ENV_NAME}
# check that the python version matches the desired one; exit immediately if not
PYVER=`python -c "from __future__ import print_function; import sys; print('{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor))"`
if [[ $PYVER != $PYTHON ]]; then
  exit 1;
fi
