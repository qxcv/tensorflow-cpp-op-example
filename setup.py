#!/usr/bin/env python3

import os

from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

__version__ = '0.0.1'
PACKAGE = 'tf_inner_product'
PACKAGE_NAME = 'tf-inner-product'
# This will only work with TF<1.14 pending
# https://github.com/tensorflow/tensorflow/issues/29643
TF_REQUIRE = 'tensorflow>=1.13.0<1.14.0'


def _build_ext(ext_obj):
    """Build C/C++ implementation of inner product ops."""
    import tensorflow as tf
    compiler = new_compiler(compiler=None,
                            dry_run=ext_obj.dry_run,
                            force=ext_obj.force)
    customize_compiler(compiler)
    compiler.add_include_dir(tf.sysconfig.get_include())
    objects = compiler.compile(
        [os.path.join(PACKAGE, '_inner_product_impl.cc')],
        debug=True,
        extra_preargs=['-std=c++11', *tf.sysconfig.get_compile_flags()])
    dest = os.path.join(PACKAGE, '_inner_product_impl.so')
    compiler.link(compiler.SHARED_LIBRARY,
                  objects,
                  dest,
                  debug=True,
                  extra_postargs=[
                      '-lstdc++', '-Wl,--no-undefined',
                      *tf.sysconfig.get_link_flags()
                  ])
    # cleanup: remove object files
    for obj in objects:
        os.unlink(obj)


class new_build_py(build_py):
    def run(self):
        _build_ext(self)
        super().run()


class new_develop(develop):
    def run(self):
        _build_ext(self)
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # create OS-specific wheels (per example TF op project on Github)
        return True


setup(
    name=PACKAGE_NAME,
    version=__version__,
    description='TensorFlow inner product demo',
    packages=[PACKAGE],
    # putting this in setup_requires should ensure that we can import tf in
    # _build_ext during setup.py execution; putting it in install_requires
    # ensures that we also have tf at run time
    setup_requires=[TF_REQUIRE],
    install_requires=[TF_REQUIRE],
    # include_package_data=True, combined with our MANIFEST.in, ensures that
    # .so files are included
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass={
        # we override both build_py and develop so that our compilation code
        # gets called both when we do "pip install -e ." and when we do "pip
        # install ." (or "pip install <name-of-package>" or whatever for the
        # current package)
        'build_py': new_build_py,
        'develop': new_develop,
    },
    zip_safe=False)
