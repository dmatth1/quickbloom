CXX ?= g++
CXXFLAGS = -std=c++20 -O3 -mavx2 -mbmi2 -mfma -fno-plt -Wall -Wextra
LDFLAGS =

SRC = src/xxh64.cpp src/parquet_bloom.cpp src/bench.cpp
OBJ = $(SRC:.cpp=.o)

bench: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f bench $(OBJ)

.PHONY: clean
