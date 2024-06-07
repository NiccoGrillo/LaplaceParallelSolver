

CXX       = mpic++
CXXFLAGS ?= -std=c++20 -Wno-suggest-override -Wno-cast-function-type
CPPFLAGS ?= -O3 -Wall -pedantic -I./../include -I${PACS_ROOT}/include -I${mkEigenInc} 
LDFLAGS ?= -L${PACS_ROOT}/lib


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
	$(CXX) $(CPPFLAGS) $(CXXFLAGS)  -c $< -o $@

# Creating object files for headers
$(OBJ_DIR)/%.o: $(INC_DIR)/%.hpp | $(OBJ_DIR)
	$(CXX)  $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
