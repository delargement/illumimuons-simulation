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
CMAKE_SOURCE_DIR = /home/spongiforma/source/geant4-example_muon/source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/spongiforma/source/geant4-example_muon/build

# Utility rule file for MU.

# Include any custom commands dependencies for this target.
include CMakeFiles/MU.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MU.dir/progress.make

CMakeFiles/MU: mu

MU: CMakeFiles/MU
MU: CMakeFiles/MU.dir/build.make
.PHONY : MU

# Rule to build all files generated by this target.
CMakeFiles/MU.dir/build: MU
.PHONY : CMakeFiles/MU.dir/build

CMakeFiles/MU.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MU.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MU.dir/clean

CMakeFiles/MU.dir/depend:
	cd /home/spongiforma/source/geant4-example_muon/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/spongiforma/source/geant4-example_muon/source /home/spongiforma/source/geant4-example_muon/source /home/spongiforma/source/geant4-example_muon/build /home/spongiforma/source/geant4-example_muon/build /home/spongiforma/source/geant4-example_muon/build/CMakeFiles/MU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MU.dir/depend

