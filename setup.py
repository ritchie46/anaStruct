import sys
from setuptools import setup

try:
    from Cython.Build import cythonize
    em = cythonize(["anastruct/cython/cbasic.pyx", "anastruct/fem/cython/celements.pyx"])
except ImportError:
    em = []

if sys.version_info[0] == 3 and sys.version_info[1] < 5:
    sys.exit('Sorry, Python < 3.5 is not supported')

setup(
    name='anastruct',
    version="1.0b2",
    description='structural engineering package',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='https://ritchievink.com',
    download_url="https://github.com/ritchie46/anaStruct",
    license='GPL-3.0',
    packages=['anastruct', 'anastruct.fem', "anastruct.fem.examples",
              "anastruct.material", "anastruct.cython", "anastruct.fem.cython"],
    install_requires=[
        "matplotlib",
        "numpy",
        "plotly"
    ],
    ext_modules=em

)
