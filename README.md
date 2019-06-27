# Example of Tensorflow Operation in C++

This repository contains an example of a simple Tensorflow operation and its gradient both implemented in C++, as described in [this article](http://davidstutz.de/implementing-tensorflow-operations-in-c-including-gradients/). It has been modified so that it can be built with `setuptools`/`distutils` instead of CMake.

## Building

Building should be as simple as `pip install .` from the current directory (or `pip install -e .` for a development copy, or whatever). See `inner_product_tests.py` for usage examples. Note that this _only_ works with TF versions prior to 1.14, pending resolution of [TF issue #29643](https://github.com/tensorflow/tensorflow/issues/29643).

## License

Copyright (c) 2016 David Stutz

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
