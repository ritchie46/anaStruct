from setuptools import setup


def read_requirements(file):
    with open(file, encoding="UTF-8") as f:
        return f.read().splitlines()


def read_file(file):
    with open(file, encoding="UTF-8") as f:
        return f.read()


try:
    from Cython.Build import cythonize

    em = cythonize(
        ["anastruct/cython/cbasic.pyx", "anastruct/fem/cython/celements.pyx"]
    )
except Exception:  # pylint: disable=broad-except
    em = []


long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")
plot_requirements = read_requirements("plot_requirements.txt")
test_requirements = read_requirements("test_requirements.txt")
__version__ = "0"
exec(read_file("anastruct/_version.py"))  # pylint: disable=exec-used

setup(
    name="anastruct",
    version=__version__,
    description="analyse 2D structures.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Ritchie Vink",
    author_email="ritchie46@gmail.com",
    url="https://ritchievink.com",
    download_url="https://github.com/ritchie46/anaStruct",
    license="GPL-3.0",
    packages=[
        "anastruct",
        "anastruct.fem",
        "anastruct.fem.system_components",
        "anastruct.fem.examples",
        "anastruct.material",
        "anastruct.cython",
        "anastruct.fem.cython",
        "anastruct.fem.plotter",
        "anastruct.fem.util",
        "anastruct.sectionbase",
    ],
    package_data={"anastruct.sectionbase": ["data/*.xml"]},
    install_requires=requirements,
    extras_require={"plot": plot_requirements, "test": test_requirements},
    ext_modules=em,
)
