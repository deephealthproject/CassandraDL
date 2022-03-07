# Copyright 2021 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
from getpass import getpass
import isic_common


def run(args):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass("Insert Cassandra user: ")
        cass_pass = getpass("Insert Cassandra password: ")

    src_dir = args.src_dir
    jobs = isic_common.get_jobs(src_dir)
    isic_common.save_images(cassandra_ip, cass_user, cass_pass)(jobs)


# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        metavar="DIR",
        required=True,
        help="Specifies the input directory for ISIC classification 2018",
    )
    run(parser.parse_args())
