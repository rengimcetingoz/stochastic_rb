from setuptools import setup

setup(
    name='Risk Budgeting library',
    url='https://github.com/rengo-python/risk_budgeting',
    author='Adil Rengim Cetingoz',
    author_email='rengimcetingoz@gmail.com',
    packages=['risk_budgeting'],
    install_requires=['numpy', 'scipy'],
    version='0.6',
    license='',
    description='A package that allows to find the risk budgeting portfolio for different risk measures given a sample of asset returns using stochastic gradient descent.',
    long_description=open('README.md').read(),
)
