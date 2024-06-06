# Compiler and flags
CXX = mpic++
CXXFLAGS = -I "../include" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/include/**" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/include/c++/11.2.0" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/base/include" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/lis/2.0.30/include" \
           -I "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include" \
           -I "$(PACS_ROOT)/src/Utilities/" \
           -O2 -std=c++20 -Wno-deprecated-enum-enum-conversion

SRC_DIR = .
OBJ_DIR = ../obj
INC_DIR = ../include

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(INC_DIR)/*.hpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET = solver

# All Target
all: $(TARGET)

# Linking
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

# Creating object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Creating object files for headers
$(OBJ_DIR)/%.o: $(INC_DIR)/%.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
