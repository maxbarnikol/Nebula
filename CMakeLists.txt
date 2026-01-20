cmake_minimum_required(VERSION 3.18)
project(nebula C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

option(NEBULA_BUILD_TESTS "Build Nebula tests" OFF)

set(NEBULA_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
	set(NEBULA_IS_TOP_LEVEL ON)
endif()
option(NEBULA_BUILD_EXECUTABLES "Build Nebula command line executables" ${NEBULA_IS_TOP_LEVEL})

set(_NEBULA_DEFAULT_ALLOW_UNSUPPORTED_COMPILER OFF)
if(WIN32)
	set(_NEBULA_DEFAULT_ALLOW_UNSUPPORTED_COMPILER ON)
endif()
option(NEBULA_CUDA_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to NVCC" ${_NEBULA_DEFAULT_ALLOW_UNSUPPORTED_COMPILER})
option(NEBULA_CUDA_ALLOW_STL_MISMATCH "Pass -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH to NVCC" OFF)


# Find CUDA, if available
if(NOT CMAKE_CUDA_COMPILER)
	# Help CMake locate nvcc when it is not on PATH (common on Windows).
	find_package(CUDAToolkit QUIET)
	if(CUDAToolkit_FOUND)
		if(WIN32)
			set(_NEBULA_NVCC "${CUDAToolkit_BIN_DIR}/nvcc.exe")
		else()
			set(_NEBULA_NVCC "${CUDAToolkit_BIN_DIR}/nvcc")
		endif()
		if(EXISTS "${_NEBULA_NVCC}")
			set(CMAKE_CUDA_COMPILER "${_NEBULA_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
		endif()
	endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)

	if(CMAKE_CUDA_COMPILER_VERSION LESS 9.2)
		message(STATUS "Compiling without CUDA support: version 9.2 or higher required.")
		set(USE_CUDA OFF)
	else()
		set(USE_CUDA ON)

		set(CMAKE_CUDA_STANDARD 17)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
		set(CMAKE_CUDA_EXTENSIONS OFF)
		add_definitions(-DCUDA_HEADERS_AVAILABLE=1)
		include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()
else()
	message(STATUS "Compiling without CUDA support: no compiler found")
	set(USE_CUDA OFF)
endif()


add_library(nebula_hdf5 INTERFACE)

# Prefer an in-tree HDF5 build (e.g. when Nebula is included via add_subdirectory).
if(TARGET hdf5-static AND TARGET hdf5_hl-static)
	target_link_libraries(nebula_hdf5 INTERFACE hdf5-static hdf5_hl-static)
else()
	# Find HDF5 library
	find_package(HDF5 1.8.13 REQUIRED C HL)
	target_include_directories(nebula_hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
	target_compile_options(nebula_hdf5 INTERFACE ${HDF5_DEFINITIONS})
	target_link_libraries(nebula_hdf5 INTERFACE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif()
if(WIN32)
	target_link_libraries(nebula_hdf5 INTERFACE Shlwapi)
endif()

list(APPEND LIBRARIES nebula_hdf5)

# Find MPI library if compiling against parallel HDF5
if(HDF5_IS_PARALLEL)
	find_package(MPI REQUIRED)
	include_directories(${MPI_CXX_INCLUDE_PATH})
	list(APPEND LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

# Find threading library
find_package(Threads REQUIRED)


if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
endif()

if(USE_CUDA)
	string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
	string(APPEND CMAKE_CUDA_FLAGS " -prec-sqrt=false")
	if(WIN32 AND NEBULA_CUDA_ALLOW_UNSUPPORTED_COMPILER)
		string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
	endif()
	if(WIN32 AND NEBULA_CUDA_ALLOW_STL_MISMATCH)
		string(APPEND CMAKE_CUDA_FLAGS " -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
	endif()
endif()

if(NEBULA_BUILD_TESTS)
	enable_testing()

	if(NOT NEBULA_TESTDATA_DIR)
		get_filename_component(NEBULA_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
		get_filename_component(COLORCUBE_ROOT "${NEBULA_ROOT}/../.." ABSOLUTE)
		set(NEBULA_TESTDATA_DIR "${COLORCUBE_ROOT}/testdata" CACHE PATH "Path to ColorCube testdata")
	endif()
endif()

add_subdirectory(source)
