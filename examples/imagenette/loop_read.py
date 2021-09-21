from cassandradl import CassandraDataset

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
from time import sleep

# Read Cassandra parameters
try:
    from private_data import cassandra_ip, cass_user, cass_pass
except ImportError:
    cassandra_ip = getpass("Insert Cassandra's IP address: ")
    cass_user = getpass('Insert Cassandra user: ')
    cass_pass = getpass('Insert Cassandra password: ')

# Init Cassandra dataset
ap = PlainTextAuthProvider(username=cass_user, password=cass_pass)
cd = CassandraDataset(ap, [cassandra_ip])

# Create three splits, with ratio 70, 20, 10 and balanced classes
cd.init_listmanager(table='imagenette.ids_160', id_col='patch_id',
                    partition_cols=['or_split', 'label'],
                    split_ncols=0, num_classes=10)
cd.read_rows_from_db()
cd.init_datatable(table='imagenette.data_160')
cd.split_setup(batch_size=128, split_ratios=[7,2,1], max_patches=13500)

for _ in range(5):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[0]):
        x, y = cd.load_batch()

# Create two splits using the original train/test partition
# (split_ncols=1) and loading all the images, ignoring balance
cd.init_listmanager(table='imagenette.ids_160', id_col='patch_id',
                    partition_cols=['or_split', 'label'],
                    split_ncols=1, num_classes=10)
cd.read_rows_from_db()
cd.init_datatable(table='imagenette.data_160')
cd.split_setup(batch_size=128, split_ratios=[1,1], bags=[[('train',)], [('val',)]], use_all_images=True)

for _ in range(5):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[0]):
        x, y = cd.load_batch()
        
