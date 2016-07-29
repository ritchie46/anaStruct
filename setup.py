from setuptools import setup
exec(open('StructuralEngineering/version.py').read())

setup(
    name='StructuralEngineering',
    version=__version__,
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
