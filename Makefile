CXX = mpic++
CXXFLAGS = -I "include" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/include/**" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/include/c++/11.2.0" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/base/include" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/lis/2.0.30/include" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include" \
           -I "/home/nicco/PACS/pacs-examples/Examples/src/Utilities/" \
           -O2 -std=c++11 -Wno-deprecated-enum-enum-conversion

SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET = solver

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: clean
