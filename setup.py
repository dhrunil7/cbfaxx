from setuptools import setup

setup(
    name="cbfax",
    version="0.0.1",
    description="cbf with jax",
    author="Karen Leung",
    author_email="kymleung@uw.edu",
    packages=["cbfax"],
    install_requires=[
        "jax",
        "matplotlib",
        "numpy",
    ],
)