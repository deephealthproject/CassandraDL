# Run with, e.g.,
# spark-submit --master spark://spark-master:7077 --conf spark.default.parallelism=10 --py-files cassandra_writer.py imagenette-spark.py --src-dir /tmp/imaginette2-160

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.cluster import ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from getpass import getpass
from tqdm import tqdm
from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import argparse
import cassandra
import io
import os
import uuid
from cassandra_writer import CassandraWriter


def save_images(cassandra_ip, cass_user, cass_pass):
    auth_prov = PlainTextAuthProvider(cass_user, cass_pass)

    def ret(jobs):
        cw = CassandraWriter(auth_prov, [cassandra_ip], 'imagenette.ids_160',
                             'imagenette.data_160', 'imagenette.metadata_160')
        for path, label, or_label, or_split in tqdm(jobs):
            cw.save_image(path, label, or_label, or_split)
    return(ret)


def run(args):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass('Insert Cassandra user: ')
        cass_pass = getpass('Insert Cassandra password: ')

    src_dir = args.src_dir
    jobs = []
    labels = dict()

    ln = 0  # next-label number
    for or_split in ['train', 'val']:
        sp_dir = os.path.join(src_dir, or_split)
        subdirs = [d.name for d in os.scandir(sp_dir) if d.is_dir()]
        for or_label in subdirs:
            # if label is new, assign a new number
            if (or_label not in labels):
                labels[or_label] = ln
                ln += 1
            label = labels[or_label]
            cur_dir = os.path.join(sp_dir, or_label)
            fns = os.listdir(cur_dir)
            for fn in fns:
                path = os.path.join(cur_dir, fn)
                jobs.append((path, label, or_label, or_split))
    # run spark
    conf = SparkConf()\
        .setAppName("Imagenette_160")
        #.setMaster("spark://spark-master:7077")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    par_jobs = sc.parallelize(jobs)
    par_jobs.foreachPartition(
        save_images(
            cassandra_ip,
            cass_user,
            cass_pass))


# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        metavar="DIR",
        required=True,
        help="Specifies the input directory for Imagenette")
    run(parser.parse_args())
