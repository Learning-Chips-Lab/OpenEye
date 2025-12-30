# setup.py file for the open_eye package
from setuptools import setup, find_packages
setup(
    name='open_eye',
    version='1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your package dependencies here
    ],
    author='Denis Lebold',
    description='OpenEye package for neural network hardware acceleration',
    url='https://github.com/Learning-Chips-Lab/OpenEye',
)