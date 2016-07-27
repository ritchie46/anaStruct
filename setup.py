from setuptools import setup

setup(
    name='StructuralEngineering',
    version="0.1.336",
    description='structural engineering package',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='http://pypi.python.org/pypi/StructuralEngineering/',
    license='MIT License',
    packages=['StructuralEngineering', 'StructuralEngineering.FEM'],
    install_requires=[
        "matplotlib",
        "numpy",
    ]
)
