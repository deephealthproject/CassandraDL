# Copyright 2021 CRS4
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
from getpass import getpass
import os
import unito_common


def run(args):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass('Insert Cassandra user: ')
        cass_pass = getpass('Insert Cassandra password: ')

    src_dir = args.src_dir
    for suffix in ['7000_224', '800']:
        cur_dir = os.path.join(src_dir, suffix)
        jobs = unito_common.get_jobs(cur_dir)
        unito_common.save_images(cassandra_ip, cass_user, cass_pass, suffix)(jobs)


# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        metavar="DIR",
        required=True,
        help="Specifies the input directory for UNITOPatho")
    run(parser.parse_args())
