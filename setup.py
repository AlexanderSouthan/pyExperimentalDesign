from setuptools import setup, find_packages

setup(
    name='pyExperimentalDesign',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'pandas', 'scipy', 'statsmodels', 'pyDataFitting']
)
