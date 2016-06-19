from setuptools import find_packages
from setuptools import setup

setup(name="yadll",
      version="0.0.1",
      description="Yet Another Deep Learning Lab. Ultra light Deep Learning framework based on Theano",
      author="Philippe Chavanne",
      author_email="philippe.chavanne@gmail.com",
      url="https://github.com/pchavanne/yadll",
      license="MIT",
      install_requires=['numpy', 'pandas', 'theano', 'docopt'],
      packages=find_packages()
      )
