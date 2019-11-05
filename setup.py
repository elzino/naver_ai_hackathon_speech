#nsml: nvcr.io/nvidia/pytorch:19.09-py3
from distutils.core import setup
import setuptools

setup(
    name='speech_hackathon',
    version='1.0',
    install_requires=[
        'python-Levenshtein==0.12.0',
        'librosa==0.7.0'
    ]
)
