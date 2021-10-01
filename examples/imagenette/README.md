# Imagenette Dataset

In this example we will import the [Imagenette2-360
dataset](https://github.com/fastai/imagenette) as a Cassandra dataset.

The raw files are already present in the provided [Docker
container](../../), from which the following commands can be run.

```bash
## - Create the tables
$ cd examples/imagenette/
$ /cassandra/bin/cqlsh -f create_tables.cql

## - Fill the tables with data and metadata
$ python3 imagenette_serial.py --src-dir /tmp/imagenette2-320/

## - Alternatively: fill the tables in parallel (20 jobs) with Spark
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files imagenette_common.py imagenette_spark.py --src-dir /tmp/imagenette2-320

## - Tight loop data loading test
$ python3 loop_read.py

## - Simple ResNet50 training (uses GPU as default)
$ python3 train.py

## - Optional: empty the tables
$ /cassandra/bin/cqlsh -f empty_tables.cql
$ /cassandra/bin/nodetool clearsnapshot --all -- imagenette
```
