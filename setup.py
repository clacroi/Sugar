"""File to enable a pip installation"""

import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='sugar',
    version='0.1.0',
    author='Corentin Lacroix',
    author_email='corent1.lacroix@gmail.com',
    description='(Caffe and) Sugar : Python library and tools for training Caffe models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6.8',
    install_requires=[
        'numpy>=1.16.4',
        'opencv-python>=3.4.3.18'
    ]
)