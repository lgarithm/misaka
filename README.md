# CrystalNet: A minimalist deep learning library in 1K-lines of modern C and C++.

This project is still under active and fast development. We are going to release more details once it is ready.


## Getting Started

Cmake and a modern C++ compiler (fully support C++17) are requird.

```
$ make
$ ./utils/download-mnist.sh
$ ./build/bin/mnist_slp_c
```

## Design Philosophy

### Minimalist style

The core library is written in 1K lines of C and C++, which means that it is super easy for you to know everything running underneath. No secret! This also has another implication meaning that this library is suitable for students and hackers. Students can quickly learn the essential pieces of a modern deep learning system; while hackers can quickly experiment crazy ideas in this minimal library.

### Modular and extensibility

The library has a modular core runtime. It is very easy to experiment ideas of, for example, memory management, synchronization, and matrix multiplication, with an aim of optimizing the performance bottlenecks that can occur at any stacks of a deep learning system. The library also has a C core which means that you can easily extend it with  language bindings, for example, Python, Lua, Golang and Rust.

### High-performance

TBD

## Concepts

### Agent
 An intelligent entity, which can percept, learn and react. An agent will be represented by a Lipschitz Transducer, a continuous extension of Finite State Transducer (FST).

### Tensor
A tensor is an linear algebra over commutative ring R,
A tensor has **shape**, **rank** and **dimension** (over R).

### Operator
TBD

### Model
TBD

### Layer
TBD

