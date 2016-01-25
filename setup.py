from setuptools import find_packages
from setuptools import setup

setup(name="dl",
      version="0.0.1",
      description="Deep learning ultra light framework based on Theano",
      author="Philippe Chavanne",
      author_email="philippe.chavanne@gmail.com",
      url="https://github.com/pchavanne/dl",
      license="MIT",
      install_requires=['numpy', 'theano'],
      packages=find_packages()
      )
