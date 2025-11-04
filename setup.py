#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the long description from README.md
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='easifa-core',
    version='2.0.0',
    description='EasIFA: Enzyme Active Site Inference Framework - Core Inference Module',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='EasIFA Team',
    author_email='your.email@example.com',
    url='https://github.com/wangxr0526/EasIFA2',
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    python_requires='>=3.8',
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'easifa-predict=easifa_core.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='enzyme active-site prediction bioinformatics deep-learning',
    include_package_data=True,
    zip_safe=False,
)
