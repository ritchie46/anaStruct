from distutils.core import setup

setup(
    name='StructuralEngineering',
    version="0.1.335",
    description='structural engineering package',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='http://pypi.python.org/pypi/StructuralEngineering/',
    license='MIT License',
    packages=['StructuralEngineering', 'FEM'],
    install_requires=[
        "matplotlib",
        "numpy",
    ]
)
