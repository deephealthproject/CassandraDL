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

cd.init_listmanager(table='imagenette.ids_160', id_col='patch_id',
                    partition_cols=['or_split', 'label'],
                    split_ncols=0, num_classes=10)
cd.read_rows_from_db()
cd.init_datatable(table='imagenette.data_160')
cd.split_setup(batch_size=128, split_ratios=[1], max_patches=14000)

for _ in range(10):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[0]):
        x, y = cd.load_batch()
