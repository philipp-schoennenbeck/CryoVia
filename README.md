#Installation

mkdir CryoViaInstallation
cd CryoViaInstallation

git clone https://github.com/philipp-schoennenbeck/CryoVia
git clone https://github.com/philipp-schoennenbeck/GridEdgeDetector

conda create -n cryovia python=3.9.16

conda activate cryovia

conda install -c conda-forge cudatoolkit=11.8.0

python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

### Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### Make sure you are in the cryovia folder (the folder with the setup.py file)
cd CryoVia
pip install .

### Install the GridEdgeDetector repo
cd ../GridEdgeDetector
pip install .
<!-- pip install git+https://github.com/philipp-schoennenbeck/GridEdgeDetector.git -->
