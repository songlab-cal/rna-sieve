from setuptools import setup
from os import path

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rnasieve',
    packages=['rnasieve'],
    version='0.1.4',
    license='GPL',
    description='A library for the statistical deconvolution of RNA bulk samples with single-cell references.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Justin Hong',
    author_email='justinhong@berkeley.edu',
    url='https://github.com/songlab-cal/rna-sieve',
    download_url='https://github.com/songlab-cal/archive/v_0_1_4.tar.gz',
    keywords=[
        'rna',
        'deconvolution',
        'statistics',
        'single-cell',
        'proportion',
        'bulk'],
    install_requires=[
        'numpy',
        'cvxpy',
        'scipy',
        'pandas',
        'altair',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
