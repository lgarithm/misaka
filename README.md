# Crystalnet: A fast deep learning library in 1K lines.

Crystalnet is a novel deep learning library written in modern C and C++. Its core implementatin has less than 1K lines of code. Though small, it is able to achieve high-performance comparable to giant engines such as Google TenserFlow; while exhibiting good extensibilty due to its clean, safe and modular library design.

Crystalnet is designed for students and hackers. Students can easily learn the essential pieces of a deep learning system in hours. Hackers can quickly experiment crazy ideas within Crystalnet in days. 

Crystalnet is under active development. We are going to release more details once it is ready.

## How to build

Crystalnet relies on modern C and C++ features to write concise yet performant code.
Its build process requires Cmake (>=3.9) and a contemporary C++ compiler (fully support C++17).

```
$ make
$ ./utils/download-mnist.sh
$ ./build/bin/mnist_slp_c
```