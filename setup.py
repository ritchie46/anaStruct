
import sys
from setuptools import setup

try:
    from Cython.Build import cythonize
    em = cythonize(['anastruct/cython/cbasic.pyx', 'anastruct/fem/cython/celements.pyx'])
except Exception:
    em = []

if sys.version_info[0] == 3 and sys.version_info[1] < 5:
    sys.exit('Sorry, Python < 3.5 is not supported')

setup(
    name='anastruct',
    version='1.1.0a1',
    description='analyse 2D structures.',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='https://ritchievink.com',
    download_url='https://github.com/ritchie46/anaStruct',
    license='GPL-3.0',
    packages=['anastruct', 'anastruct.fem', 'anastruct.fem.system_components', 'anastruct.fem.examples',
              'anastruct.material', 'anastruct.cython', 'anastruct.fem.cython', 'anastruct.fem.plotter',
              'anastruct.fem.util', 'anastruct.sectionbase'],
    package_data={'': ['*.xml']},
    package_dir='',
    install_requires=[
        'matplotlib>=3.0, <3.1',
        'numpy>=1.15.4',
        'scipy>=1.1.0'
    ],
    ext_modules=em

)