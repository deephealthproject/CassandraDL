// Copyright 2021 CRS4
// 
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "batchpatchhandler.hpp"

PYBIND11_MODULE(BPH, m) {
  py::class_<BatchPatchHandler>(m, "BatchPatchHandler")
    .def(py::init<int, ecvl::Augmentation*, string, string, string, string, string, string, vector<string>, int, int >(), "num_classes"_a, "aug"_a, "table"_a, "label_col"_a, "data_col"_a, "id_col"_a, "username"_a, "cass_pass"_a, "cassandra_ips"_a, "thread_par"_a=32, "port"_a=9042)
    .def("schedule_batch", &BatchPatchHandler::schedule_batch, "keys"_a)
    .def("block_get_batch", &BatchPatchHandler::block_get_batch)
    .def("ignore_batch", &BatchPatchHandler::ignore_batch);
}
