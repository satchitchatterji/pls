from setuptools import setup, find_packages

# matplotlib and torch versions have been changed from original (https://github.com/wenchiyang/pls/blob/main/setup.py) 
setup(
    name="pls",
    version="0.0.1",
    packages=find_packages(
        where='.'
    ),
    install_requires=[
        # environment
        "protobuf",
        "gym",
        "torch",
        "stable-baselines3[extra]",
        # shielding
        "problog",
        "pysdd",
        # # experiments -- dask
        # "dask[complete]",
        # "asyncssh",
        # "bokeh",
        # visialization
        "tensorboard",
        "matplotlib",
        "scikit-image"
    ],
)
