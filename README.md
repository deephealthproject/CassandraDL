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
$ /cassandra/bin/cassandra   # - wait until "state jump to NORMAL" (about 1 minute)
                             #   (note that the shell prompt is immediately returned)

## - Create the tables
$ cd examples/imagenette/
$ /cassandra/bin/cqlsh -f create_tables.cql

## - Fill the tables with data and metadata
$ python3 imagenette_serial.py --src-dir /tmp/imagenette2-320/

## - Tight loop data loading test
$ python3 loop_read.py

## - Simple VGG16 training (uses GPU as default)
$ python3 train.py

## - Empty the tables, to fill them again using Spark
$ /cassandra/bin/cqlsh -f empty_tables.cql

## - Start Spark master+worker
$ sudo /spark/sbin/start-master.sh
$ sudo /spark/sbin/start-worker.sh spark://$HOSTNAME:7077

## - Fill the tables in parallel (20 jobs) with Spark
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files imagenette_common.py imagenette_spark.py --src-dir /tmp/imagenette2-320
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


## Authors

Cassandra Data Loader is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>
  * Giovanni Busonera, CRS4 <giovanni.busonera@crs4.it>

## License

Cassandra Data Loader is licensed under the MIT License.
See LICENSE for further details.

## Acknowledgment

- Jakob Progsch for his [ThreadPool code](https://github.com/progschj/ThreadPool)
