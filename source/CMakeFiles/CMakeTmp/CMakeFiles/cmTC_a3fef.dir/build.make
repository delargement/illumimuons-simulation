# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp

# Include any dependencies generated for this target.
include CMakeFiles/cmTC_a3fef.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cmTC_a3fef.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cmTC_a3fef.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cmTC_a3fef.dir/flags.make

CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.o: CMakeFiles/cmTC_a3fef.dir/flags.make
CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.o: /usr/share/cmake/Modules/CheckFunctionExists.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.o -c /usr/share/cmake/Modules/CheckFunctionExists.c

CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.i: cmake_force
	@echo "Preprocessing C source to CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /usr/share/cmake/Modules/CheckFunctionExists.c > CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.i

CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.s: cmake_force
	@echo "Compiling C source to assembly CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /usr/share/cmake/Modules/CheckFunctionExists.c -o CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.s

# Object files for target cmTC_a3fef
cmTC_a3fef_OBJECTS = \
"CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.o"

# External object files for target cmTC_a3fef
cmTC_a3fef_EXTERNAL_OBJECTS =

cmTC_a3fef: CMakeFiles/cmTC_a3fef.dir/CheckFunctionExists.c.o
cmTC_a3fef: CMakeFiles/cmTC_a3fef.dir/build.make
cmTC_a3fef: CMakeFiles/cmTC_a3fef.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cmTC_a3fef"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmTC_a3fef.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cmTC_a3fef.dir/build: cmTC_a3fef
.PHONY : CMakeFiles/cmTC_a3fef.dir/build

CMakeFiles/cmTC_a3fef.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cmTC_a3fef.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cmTC_a3fef.dir/clean

CMakeFiles/cmTC_a3fef.dir/depend:
	cd /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp /home/spongiforma/source/geant4-example_muon/source/CMakeFiles/CMakeTmp/CMakeFiles/cmTC_a3fef.dir/DependInfo.cmake
.PHONY : CMakeFiles/cmTC_a3fef.dir/depend
