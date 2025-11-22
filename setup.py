from setuptools import setup, find_packages
from pathlib import Path


def read_requirements():
    return list(Path("requirements.txt").read_text().splitlines())

setup(
    name='gtfread',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    
    # Descriptions
    description='A microscopic library to read GTF files.',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    
    # Details
    maintainer="Alisamar Husian",
    maintainer_email="zrthxn@gmail.com",
    url="https://github.com/zrthxn/gtfread",
)
