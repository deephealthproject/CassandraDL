# Cassandra Data Loader for the DeepHealth Toolkit

## Overview

## Installation

The easiest way to test the Cassandra Data Loader is by using the
provided Dockerfile, which also contains the DeepHealth Toolkit, an
Apache Cassandra server and Apache Spark.

The details of how to install the Cassandra Data Loader in a system
which already provides the DeepHealth Toolkit can be easily deduced
from the [Dockerfile](Dockerfile).

```bash
## Build and run cassandradl docker container
$ docker build -t cassandradl .
$ docker run --rm -it --cap-add=sys_nice cassandradl

## Inside the Docker container:

## - Start Cassandra server
$ /cassandra/bin/cassandra   # Note that the shell prompt is immediately returned
                             # Wait until "state jump to NORMAL" is shown (about 1 minute)

## - Start Spark master+worker
$ sudo /spark/sbin/start-master.sh
$ sudo /spark/sbin/start-worker.sh spark://$HOSTNAME:7077
```

## Dataset examples

- [Imagenette](examples/imagenette/)
- [UNITOPatho](examples/unitopatho/)
- [ISIC Skin Lesion 2018](examples/isic_2018/)

## Requirements

Cassandra Data Loader requires:
- DeepHealth Toolkit (with EDDL and ECVL)
- Cassandra C/C++ driver
- Cassandra Python driver
- OpenCV

All the required libraries are already installed in the provided
Dockerfile.

## Documentation

In progress...

## Further details


## Authors

Cassandra Data Loader is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>
  * Giovanni Busonera, CRS4 <giovanni.busonera@crs4.it>

## License

Cassandra Data Loader is licensed under the MIT License.
See LICENSE for further details.

## Acknowledgment

- Jakob Progsch for his [ThreadPool code](https://github.com/progschj/ThreadPool)
