//////////////////////////////////////////////////////////////////////
// isic classification 2018: https://github.com/deephealthproject/use-case-pipelines

CREATE KEYSPACE isic WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

// Mapping patch metadata to data via uuid
CREATE TABLE isic.ids_224(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE isic.data_224(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE isic.metadata_224(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);
