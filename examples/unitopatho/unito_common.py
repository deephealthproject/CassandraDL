# Copyright 2021 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import tqdm
import io
import numpy as np
import os
import uuid
import pandas as pd
from cassandradl import CassandraWriter


def get_data(path):
    with open(path, "rb") as f:
        data = f.read()
    return data


def save_images(cassandra_ip, cass_user, cass_pass, suffix):
    auth_prov = PlainTextAuthProvider(cass_user, cass_pass)

    def ret(jobs):
        cols = [
            "image_id",
            "top_label_name",
            "type_name",
            "type",
            "grade_name",
            "grade",
            "wsi",
            "roi",
            "mpp",
            "x",
            "y",
            "w",
            "h",
            "or_split",
        ]
        cw = CassandraWriter(
            auth_prov,
            [cassandra_ip],
            table_ids=f"unito.ids_{suffix}",
            table_data=f"unito.data_{suffix}",
            table_metadata=f"unito.metadata_{suffix}",
            id_col="patch_id",
            label_col="top_label",
            data_col="data",
            cols=cols,
            get_data=get_data,
        )
        for path, label, partition_items in tqdm(jobs):
            cw.save_image(path, label, partition_items)

    return ret


def get_jobs(src_dir):
    jobs = []
    labels = dict()
    ln = 0  # next-label number
    for or_split in ["train", "test"]:
        csv = os.path.join(src_dir, or_split + ".csv")
        tab = pd.read_csv(csv, keep_default_na=False)
        for _, r in tab.iterrows():
            cur_dir = os.path.join(src_dir, r["top_label_name"])
            path = os.path.join(cur_dir, r["image_id"])
            partition_items = r.to_list() + [or_split]
            label = partition_items.pop(2)
            jobs.append((path, label, partition_items))
    return jobs
