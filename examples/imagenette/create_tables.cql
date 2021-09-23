//////////////////////////////////////////////////////////////////////
// imagenette: https://github.com/fastai/imagenette

CREATE KEYSPACE imagenette WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

// Mapping patch metadata to data via uuid
CREATE TABLE imagenette.ids_160(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenette.data_160(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenette.metadata_160(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);
