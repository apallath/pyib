from setuptools import setup, find_packages
import pathlib
import pkg_resources

version = "0.0.1"

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    requirements = [
        str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(name='pyib',
      version=version,
      description=(
          """PyTorch & OpenMM implementations of information bottleneck based approaches for
          learning and biasing reaction coordinates in molecular simulations."""
      ),
      long_description=readme,
      long_description_content_type='text/markdown',
      author='Yusheng Cai, Akash Pallath',
      author_email='cys9741@seas.upenn.edu, apallath@seas.upenn.edu',
      url='https://github.com/apallath/pyib',
      python_requires='>=3.6',
      install_requires=requirements,
      packages=find_packages())
