# Cassandra Data Loader for the DeepHealth Toolkit

## Overview

CassandraDL is a data loader for the DeepHealth toolkit, which
leverages [Apache Cassandra NoSQL DB](https://cassandra.apache.org/)
for storing both data and metadata, making them efficiently available
through the network, and allowing automatic split creation and easier
data distribution.

## Installation

The easiest way to test the Cassandra Data Loader is by using the
provided Dockerfile, which also contains the DeepHealth Toolkit, an
Apache Cassandra server and Apache Spark.

The details of how to install the Cassandra Data Loader in a system
which already provides the DeepHealth Toolkit can be easily deduced
from the [Dockerfile](Dockerfile).

For better performance and for data persistence, it is strongly
advised to mount a host directory for Cassandra on a fast disk (e.g.,
`/mnt/fast_disk/cassandra`), as shown in the commands below.

```bash
## Build and run cassandradl docker container
$ docker build -t cassandradl .
$ docker run --rm -it -v /mnt/fast_disk/cassandra:/cassandra/data:rw --cap-add=sys_nice cassandradl

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

Documentation (in progress...) for the latest CassandraDL version is
available via
[ReadTheDocs](https://cassandradl.readthedocs.io/en/latest/).

### Generation via Sphinx

Alternatively, documentation can be generated locally via Sphinx.

To install Sphinx:
```bash
$ pip3 install sphinx sphinx_rtd_theme sphinx-autoapi
```

To generate the html documentation (accessible at location `docs/build/html/index.html`):
```bash
$ make -C docs/ html
```

## Further details

An article describing the architecture and performance of CassandraDL
has been presented at [IEEE BigData
2021](http://bigdataieee.org/BigData2021/), in the Special Session on
*Machine Learning on Big Data*. It is available either via
[IEEE Xplore](https://ieeexplore.ieee.org/document/9672005) or
the CRS4 publications repository [(direct link to
PDF)](http://publications.crs4.it/pubdocs/2021/VB21/cassandra-ml.pdf).

### Citation

```bibtex
@InProceedings{cassandradl,
  author       = {Versaci, F. and Busonera, G.},
  title        = {Scaling deep learning data management with Cassandra DB},
  booktitle    = {2021 IEEE International Conference on Big Data (Big Data)},
  month        = {december},
  year         = {2021},
  doi          = {10.1109/BigData52589.2021.9672005},
  isbn         = {978-1-6654-3902-2},
}
```

## Authors

Cassandra Data Loader is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>
  * Giovanni Busonera, CRS4 <giovanni.busonera@crs4.it>

## License

Cassandra Data Loader is licensed under the MIT License.
See LICENSE for further details.

## Acknowledgment

- Jakob Progsch for his [ThreadPool code](https://github.com/progschj/ThreadPool)
