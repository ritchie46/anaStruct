import sys
from setuptools import setup
exec(open('StructuralEngineering/version.py').read())

if sys.version_info[0] == 3 and sys.version_info[1] < 5:
    sys.exit('Sorry, Python < 3.5 is not supported')

setup(
    name='StructuralEngineering',
    version=__version__,
    description='structural engineering package',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='https://ritchievink.com',
    license='MIT License',
    packages=['StructuralEngineering', 'StructuralEngineering.FEM', "StructuralEngineering.FEM.examples",
              "StructuralEngineering.FEM.material"],
    install_requires=[
        "matplotlib",
        "numpy",
    ]
)
