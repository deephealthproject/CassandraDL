import cassandra
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.cluster import ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
import uuid


class CassandraWriter():
    def __init__(self, auth_prov, cassandra_ips, table1, table2,
                 table3, get_data):
        self.get_data = get_data
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

    def save_image(self, path, label, or_label, or_split):
        # read file into memory
        data = self.get_data(path)
        patch_id = uuid.uuid4()
        item = patch_id, label, data, or_label, or_split
        self.save_item(item)

