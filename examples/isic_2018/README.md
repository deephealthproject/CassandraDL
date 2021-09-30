# ISIC Skin Lesion Classification 2018 Dataset

In this example we will import the [ISIC skin lesion classification
2018 dataset](https://challenge.isic-archive.com/landing/2018/47) as a
Cassandra dataset.

The dataset (3 GB) needs to be downloaded from the [DeepHealth
use-cases
page](https://github.com/deephealthproject/use-case-pipelines), before
running the following commands (from the provided [Docker
container](../../)), assuming `/data/isic_classification_2018/` as
source directory.

```bash
## - Create the tables
$ cd examples/isic_2018/
$ /cassandra/bin/cqlsh -f create_tables.cql

## - Fill the tables with data and metadata
$ python3 isic_serial.py --src-dir /data/isic_classification_2018/

## - Alternatively: fill the tables in parallel (20 jobs) with Spark
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files isic_common.py isic_spark.py --src-dir /data/isic_classification_2018/

## - Tight loop data loading test
$ python3 loop_read.py

## - Simple ResnNet50 training (uses GPU as default)
$ python3 train.py

## - Optional: empty the tables
$ /cassandra/bin/cqlsh -f empty_tables.cql
```
