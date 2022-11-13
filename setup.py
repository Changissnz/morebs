from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='morebs2',
    version='0.0.2',
    description='data structures to aid in numerical data generation and clustering',
    long_description=readme,
    author='Richard Pham',
    author_email='phamrichard45@gmail.com',
    url='https://github.com/Changissnz/morebs2',
    #license=license,
    packages=find_packages(exclude=('tests','docs'))
)
