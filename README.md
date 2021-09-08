# Cassandra Data Loader for EDDL

## Overview

## Installation

The easiest way to run the Cassandra Data Loader together with the
DeepHealth Toolkit is by using the provided Dockerfile, i.e.:

```bash
$ docker build -t cassandradl .
$ docker run --rm -it cassandradl
# python3 examples/loop_read.py
```

## Requirements

Cassandra Data Loader requires:
- DeepHealth Toolkit (with EDDL and ECVL)
- Cassandra C/C++ driver
- Cassandra Python driver
- OpenCV

All the required libraries are already installed in the provided
Dockerfile.

## Documentation


## Further details


## Author

`CassandraDL` is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>
  * Giovanni Busonorea, CRS4 <giovanni.busonera@crs4.it>

## License

Cassandra Data Loader is licensed under the MIT License.
See LICENSE for further details.

## Acknowledgment

- Jakob Progsch for his [ThreadPool code](https://github.com/progschj/ThreadPool)
