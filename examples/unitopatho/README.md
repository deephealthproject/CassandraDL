# UNITOPatho Dataset

In this example we will import the [UNITOPatho
dataset](https://ieee-dataport.org/open-access/unitopatho) as a
Cassandra dataset.

The original dataset (275 GB) needs to be downloaded and unzipped
before running the following commands (from the provided [Docker
container](../../)), assuming `/data/unitopath-public/` as source
directory.

```bash
## - Create the tables
$ cd examples/unitopatho/
$ /cassandra/bin/cqlsh -f create_tables.cql

## - Fill the tables with data and metadata
$ python3 unito_serial.py --src-dir /data/unitopath-public/

## - Alternatively: fill the tables in parallel (20 jobs) with Spark
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files unito_common.py unito_spark.py --src-dir /data/unitopath-public/

## - Tight loop data loading test
$ python3 loop_read.py

## - Simple ResnNet50 training (uses GPU as default)
$ python3 train.py

## - Optional: empty the tables
$ /cassandra/bin/cqlsh -f empty_tables.cql
```
