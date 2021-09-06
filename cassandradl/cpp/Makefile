# This Makefile is for debug purposes only
# Compilation is done automatically when running "pip3 install ." using setup.py

CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O2
LFLAGS = -lcassandra -lopencv_core -lopencv_imgcodecs -leddl \
-lecvl_core -lecvl_eddl $(shell python3-config --ldflags)
BIND_INCL = $(shell python3 -m pybind11 --includes)
BIND_SUFF = $(shell python3-config --extension-suffix)
IXXFLAGS = -I/usr/local/include/eigen3/ -I/usr/include/opencv4 $(BIND_INCL)

.PHONY: clean all

all: BPH

clean:
	rm -f *.o runme BPH$(BIND_SUFF)

BPH: pybindings.cpp batchpatchhandler.cpp
	g++ $(CXXFLAGS) $(IXXFLAGS) -shared -fPIC $^ -o $@$(BIND_SUFF) $(LFLAGS)

%.o : %.cpp
	g++ $(CXXFLAGS) $(IXXFLAGS) -c -o $@ $< $(LFLAGS)

dep.mk: *.cpp *.hpp
	g++ -MM *.cpp > dep.mk

include dep.mk
