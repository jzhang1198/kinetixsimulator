from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='kinetixsimulator',
    version='0.1.0',
    packages=find_packages(),
    author= 'Jonathan Zhang',
    author_email="jon.zhang@ucsf.edu",
    description='Python package for interactive simulation of kinetic mechanisms.',
    url='https://github.com/jzhang1198/kinetixsimulator',
    classifiers=[
        'Programming Language :: Python :: 3.9.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=requirements,
)