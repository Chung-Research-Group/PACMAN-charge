import requests
from setuptools import setup, find_packages

setup(
    name="PACMAN-charge",
    version="1.3.9",
    packages=find_packages(),
    description="Partial Atomic Charges for Porous Materials based on Graph Convolutional Neural Network (PACMAN)",
    author="Guobin Zhao",
    author_email="sxmzhaogb@gmai.com",
    url="https://github.com/mtap-research/PACMAN-charge/",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
        'PACMANCharge': ['*.json']
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "requests",
        "numpy>=1.13.3",
        "pymatgen>=2018.6.11",
        "ase>=3.19",
        "tqdm>=4.15",
        "pandas>=0.20.3",
        "scikit-learn>=0.19.1",
        "joblib>= 0.13.2",
        "torch",
        "PyCifRW==4.4.6"
    ],
)
