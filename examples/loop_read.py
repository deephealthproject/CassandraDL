from cassandradl import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
from time import sleep

## Read Cassandra ip and password
try:
    from private_data import inet_pass, cassandra_ip
except ImportError:
    cassandra_ip = getpass("Insert Cassandra's IP address: ")
    inet_pass = getpass('Insert Cassandra password: ')

## Init Cassandra dataset
ap = PlainTextAuthProvider(username='inet', password=inet_pass)
cd = CassandraDataset(ap, [cassandra_ip])

cd.init_listmanager(table='imagenet.ids_224', id_col='patch_id',
                    split_ncols=0, num_classes=1000,
                    partition_cols=['label'])
cd.read_rows_from_db()
cd.init_datatable(table='imagenet.data_224')
cd.split_setup(batch_size=128, split_ratios=[1], max_patches=130000)

for _ in range(10):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[0]):
        x,y = cd.load_batch()
