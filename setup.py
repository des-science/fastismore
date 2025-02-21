from setuptools import setup

setup(
    name='fastismore',
    version='0.3.0',
    description='Fast importance sampling for model robustness evaluation.',
    url='https://github.com/des-science/fastismore',
    author='Otavio Alves',
    author_email='oalves@umich.edu',
    license='',
    packages=['fastismore'],
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'matplotlib',
        'getdist',
        ],
    scripts=[
        'bin/fastis-extract-ini',
        'bin/fastis-prune-chain',
        'bin/fastis-polychord2cosmosis',
        ],
    entry_points = {
        'console_scripts': [
            'fastis-plot=fastismore.plot:main',
            'fastis-sample=fastismore.sample:main',
            ],
    },
    classifiers=[]
)
