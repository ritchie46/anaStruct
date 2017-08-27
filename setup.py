import sys
from setuptools import setup
exec(open('anastruct/version.py').read())

if sys.version_info[0] == 3 and sys.version_info[1] < 5:
    sys.exit('Sorry, Python < 3.5 is not supported')

setup(
    name='anastruct',
    version=__version__,
    description='structural engineering package',
    author='Ritchie Vink',
    author_email='ritchie46@gmail.com',
    url='https://ritchievink.com',
    license='MIT License',
    packages=['anastruct', 'anastruct.fem', "anastruct.fem.examples",
              "anastruct.material"],
    install_requires=[
        "matplotlib",
        "numpy",
        "plotly"
    ]
)
