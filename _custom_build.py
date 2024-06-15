from Cython.Build import cythonize
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self) -> None:
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self) -> None:
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        cythonize(["anastruct/cython/basic.py", "anastruct/fem/cython/elements.py"])

        self.distribution.ext_modules.append(
            Extension(
                "anastruct.fem.cython",
                sources=["anastruct/cython/basic.c", "anastruct/fem/cython/elements.c"],
            )
        )
