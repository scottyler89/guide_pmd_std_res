from setuptools import setup, find_packages

# Read the content of your requirements.txt file
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='guide_pmd_std_res',
    version='0.1.0',
    #packages=find_packages(exclude=('tests*', 'testing*')),
    #license='TBD',
    description='A package for analyzing CRISPR screens (or similar data), using PMD standardized residuals with linear modeling downstream',
    long_description=open('README.md').read(),
    install_requires=required_packages,
    url='https://github.com/scottyler89/guide_pmd_std_res',
    author='Scott Tyler',
    author_email='scottyler89@gmail.com'
)


