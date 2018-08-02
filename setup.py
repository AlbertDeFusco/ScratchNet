from setuptools import setup, find_packages
from version import find_version

setup(
        name = 'ScratchNet',
        author = 'Albert DeFusco',
        description = 'Neural network from scratch in NumPy',
        license = 'MIT',
        version = find_version('scratchnet', '__init__.py'),
        packages = find_packages()
     )

