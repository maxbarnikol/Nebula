add_subdirectory(io)
add_subdirectory(geometry/octree)

if(USE_CUDA)
	add_library(nebula_gpu_lib STATIC gpu_simulation.cu
		$<TARGET_OBJECTS:io>
		$<TARGET_OBJECTS:octree>)
	target_include_directories(nebula_gpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(nebula_gpu_lib PUBLIC ${LIBRARIES})

	add_executable(nebula_gpu gpu_main.cu)
	target_link_libraries(nebula_gpu nebula_gpu_lib ${LIBRARIES})

	if(NEBULA_BUILD_TESTS)
		add_executable(nebula_gpu_lib_tests tests/nebula_gpu_lib_tests.cpp)
		target_link_libraries(nebula_gpu_lib_tests nebula_gpu_lib ${LIBRARIES})
		target_include_directories(nebula_gpu_lib_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
		add_test(NAME nebula_gpu_lib_tests
			COMMAND nebula_gpu_lib_tests ${NEBULA_TESTDATA_DIR})
	endif()
endif()

add_executable(nebula_cpu_mt cpu_mt_main.cpp
		$<TARGET_OBJECTS:io>
		$<TARGET_OBJECTS:octree>)
add_executable(nebula_cpu_edep cpu_energydep.cpp
		$<TARGET_OBJECTS:io>
		$<TARGET_OBJECTS:octree>)
target_link_libraries(nebula_cpu_mt ${LIBRARIES} Threads::Threads)
target_link_libraries(nebula_cpu_edep ${LIBRARIES} Threads::Threads)
