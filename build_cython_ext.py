from Cython.Distutils import build_ext

# from Cython.Build import cythonize
from setuptools import Extension

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

ext_modules = [
    Extension(
        "anastruct.fem.cython",
        sources=["anastruct/cython/cbasic.pyx", "anastruct/fem/cython/celements.pyx"],
        define_macros=define_macros,
    ),
]


class BuildExt(build_ext):
    def build_extensions(self) -> None:
        for ext in ext_modules:
            self.build_extension(ext)


def build(setup_kwargs: dict) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": BuildExt},
        }
    )
