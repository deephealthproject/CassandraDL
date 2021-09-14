from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.cluster import ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from getpass import getpass
import argparse
import cassandra
import numpy as np
import io
import os
import uuid


class CassandraWriter():
    def __init__(self, auth_prov, cassandra_ips, table1, table2,
                 table3):
        prof = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory)
        profs = {'default': prof}
        self.cluster = Cluster(cassandra_ips,
                               execution_profiles=profs,
                               protocol_version=4,
                               auth_provider=auth_prov)
        self.sess = self.cluster.connect()
        query1 = f"INSERT INTO {table1} "\
            + f"(label, or_label, or_split, patch_id) VALUES (?,?,?,?)"
        query2 = f"INSERT INTO {table2} "\
            + "(patch_id, label, data) VALUES (?,?,?)"
        query3 = f"INSERT INTO {table3} "\
            + f"(label, or_label, or_split, patch_id) VALUES (?,?,?,?)"
        self.prep1 = self.sess.prepare(query1)
        self.prep2 = self.sess.prepare(query2)
        self.prep3 = self.sess.prepare(query3)

    def __del__(self):
        self.cluster.shutdown()

    def save_item(self, item):
        # if buffer full pop two elements from top
        patch_id, label, data, or_label, or_split = item
        i1 = self.sess.execute_async(self.prep1, (label,
                                                  or_label,
                                                  or_split,
                                                  patch_id),
                                     execution_profile='default',
                                     timeout=30)
        i3 = self.sess.execute_async(self.prep3, (label,
                                                  or_label,
                                                  or_split,
                                                  patch_id),
                                     execution_profile='default',
                                     timeout=30)
        # wait for remaining async inserts to finish
        i1.result()
        i3.result()
        # insert heavy data synchronously
        self.sess.execute(self.prep2, (patch_id, label, data),
                          execution_profile='default', timeout=30)

    def get_data(self, path):
        img = Image.open(path).convert('RGB')
        # resize and crop to 160x160
        tg = 160
        sz = np.array(img.size)
        min_d = sz.min()
        sc = float(tg) / min_d
        new_sz = (sc * sz).astype(int)
        img = img.resize(new_sz)
        off = (new_sz.max() - tg) // 2
        if (new_sz[0] > new_sz[1]):
            box = [off, 0, off + tg, tg]
        else:
            box = [0, off, tg, off + tg]
        img = img.crop(box)
        # save to stream
        out_stream = io.BytesIO()
        img.save(out_stream, format='JPEG')
        # write to db
        out_stream.flush()
        data = out_stream.getvalue()
        return(data)

    def save_image(self, path, label, or_label, or_split):
        # read file into memory
        data = self.get_data(path)
        patch_id = uuid.uuid4()
        item = patch_id, label, data, or_label, or_split
        self.save_item(item)

