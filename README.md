# Usage
For usage see [Usage](How_to_use.md)

# Installation

## Linux

### Creating directories
```mkdir CryoViaInstallation```

```cd CryoViaInstallation```

```git clone https://github.com/philipp-schoennenbeck/CryoVia```

```git clone https://github.com/philipp-schoennenbeck/GridEdgeDetector```

### Creating python environment
```conda create -n cryovia python=3.9.16```

```conda activate cryovia```

### Installing tensorflow 

```conda install -c conda-forge cudatoolkit=11.8.0```

```python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*```

```mkdir -p $CONDA_PREFIX/etc/conda/activate.d```

```echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```

```echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```

```source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```

### Verifying tensorflow install
```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```

### Installing CryoVia
```cd CryoVia```

```pip install .```

### Install GridEdgeDetector
```cd ../GridEdgeDetector```

```pip install .```


## Windows

### Creating directories
```mkdir CryoViaInstallation```

```cd CryoViaInstallation```

```git clone https://github.com/philipp-schoennenbeck/CryoVia```

```git clone https://github.com/philipp-schoennenbeck/GridEdgeDetector```

### Creating python environment
```conda create -n cryovia python=3.9.16```

```conda activate cryovia``

### Installing tensorflow

```conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0```

```pip install tensorflow==2.10```

```pip install numpy==1.21.6```

### Verifying tensorflow install

```python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```

### Installing CryoVia
```cd CryoVia```

```pip install .```

### Install GridEdgeDetector
```cd ../GridEdgeDetector```

```pip install .```
