MAKEFILEH_DIR=$(PACS_ROOT)
#
DEBUG=no #get full optimization
include $(MAKEFILEH_DIR)/Makefile.inc
#
# You may have an include file also in the current directory
# This is optional. If not present is not an error
-include Makefile.inc

#
# The general setting is as follows:
# mains are identified bt main_XX.cpp
# all other files are XX.cpp
#

# get all files *.cpp
SRCS=$(wildcard *.cpp)
# get the corresponding object file
OBJS = $(SRCS:.cpp=.o)
# get all headers in the working directory
HEADERS=$(wildcard *.hpp)
#
OBJS_NOPY = $(filter-out py%.o,$(OBJS))#
PY_OBJS = $(filter py%.o,$(OBJS))
exe_sources=$(filter main%.cpp,$(SRCS))
EXEC=$(exe_sources:.cpp=)
OBJS_NOEXEC=$(filter-out main%.o,$(OBJS))

#thi is where JacobiSolver.hpp is stored
CPPFLAGS+=-I./../include

MODULENAME=solver# Must be consistsnt with what declared in the pybind wrapper
PY_INCLUDES != python3-config --includes
PY_EXT != python3-config --extension-suffix
PY_FLAGS=-fPIC -flto #flto is not strictly needed
PY_MODULE=$(MODULENAME)$(PY_EXT)
CXXFLAGS+=$(PY_FLAGS)
CPPFLAGS+=$(PY_INCLUDES)
LDFLAGS+=-flto

LDLIBS += -lmpi # Add MPI library

####### now the definitions of the targets
#========================== NEW THE DEFINITION OF THE TARGETS
# Directories
SRC_DIR = ../src
OBJ_DIR = ../obj
INC_DIR = ../include
PYBIND_DIR = .



# Get all files *.cpp in pybind directory
PY_SRCS = $(wildcard $(PYBIND_DIR)/*.cpp)
# Get the corresponding object file
PY_OBJS = $(PY_SRCS:.cpp=.o)
# Include the source files from the src directory
SRC_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
SRC_OBJS = $(SRC_SRCS:.cpp=.o)

####### Now the definitions of the targets
.PHONY: all clean distclean

.DEFAULT_GOAL = all

all: $(PY_MODULE)

$(PY_MODULE): $(PY_OBJS) $(SRC_OBJS)
	$(RM) ./*.so
	$(CXX) $(CPPFLAGS) -shared $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $(PY_MODULE)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(PYBIND_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	$(RM) -f $(PY_OBJS) $(SRC_OBJS) $(PY_MODULE)

distclean:
	$(MAKE) clean
	$(RM) *.so *.bak *~