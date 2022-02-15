# Copyright 2021 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Run with, e.g.,
# /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files imagenette_common.py imagenette_spark.py --src-dir /tmp/imagenette2-320

import argparse
from getpass import getpass
import imagenette_common
from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


def run(args):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass("Insert Cassandra user: ")
        cass_pass = getpass("Insert Cassandra password: ")

    src_dir = args.src_dir
    jobs = imagenette_common.get_jobs(src_dir)
    # run spark
    conf = SparkConf().setAppName("Imagenette_224")
    # .setMaster("spark://spark-master:7077")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    par_jobs = sc.parallelize(jobs)
    par_jobs.foreachPartition(
        imagenette_common.save_images(cassandra_ip, cass_user, cass_pass)
    )


# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        metavar="DIR",
        required=True,
        help="Specifies the input directory for Imagenette",
    )
    run(parser.parse_args())
