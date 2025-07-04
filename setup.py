import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
# def read(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "cryovia",
    version = "0.0.2",
    author = "Philipp Schönnenbeck",
    author_email = "p.schoennenbeck@fz-juelich.de",
    description = ("cryovia"),
    license = "BSD",
    keywords = "",
    # url = "http://packages.python.org/an_example_pypi_project",
    packages=find_packages(),
    # long_description=read('README'),
    classifiers=[
        # "Development Status :: 3 - Alpha",
        # "Topic :: Utilities",
        # "License :: OSI Approved :: BSD License",
    ]
    ,
    install_requires=[
        # 'numpy==1.24.1',
        "numpy",
        'mrcfile',
        'scipy',
        'opencv-python',
        # 'sknw==0.13',
        'napari[all]==0.4.19',
        # 'matplotlib==3.8.3',
        "matplotlib",
        "pandas",
        "pyqtgraph",
        "qimage2ndarray",
        'scikit-learn==1.2.2',
        "scikit-image",
        'seaborn', 
        'sparse',
        'tqdm',
        "silence-tensorflow",
        "starfile",
        "ray",
        "circle-fit"

    ],
    entry_points={
        'console_scripts': [
            'cryovia = cryovia.gui.starting_menu:startGui',
        ]
    },
    include_package_data=True,
)