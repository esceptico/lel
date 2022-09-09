import os

import pkg_resources
from setuptools import find_packages, setup

PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
REQUIREMENTS_PATH = os.path.join(PATH_ROOT, "requirements.txt")


def parse_requirements(path: str):
    with open(path) as fp:
        text = fp.read()
    requirements = [str(r) for r in pkg_resources.parse_requirements(text)]
    return requirements


setup(
    name='lel',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/esceptico/lel',
    license='',
    author='esceptico',
    author_email='ganiev.tmr@gmail.com',
    description='Lightweight NER pipeline'
)
