from distutils.core import setup

setup(
    name='structural_engineering',
    version="0.1.1",
    description='structural engineering package',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='http://pypi.python.org/pypi/structural_engineering/',
    license='MIT License',
    packages=['StructuralEngineering', ],
    install_requires=[
        "matplotlib",
        "numpy",
    ]
)
