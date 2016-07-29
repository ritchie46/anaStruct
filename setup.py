from setuptools import setup

setup(
    name='StructuralEngineering',
    version="dev0.1",
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
