get_filename_component(Project_ID ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(Project_ID "sample_${Project_ID}")
message( STATUS "-- " ${Project_ID} )

if(USE_TBB)
    add_executable( ${Project_ID} main.cpp )
    add_dependencies(${Project_ID} ppc_onetbb)
    target_link_directories(${Project_ID} PUBLIC ${CMAKE_BINARY_DIR}/ppc_onetbb/install/lib)
    if(NOT MSVC)
        target_link_libraries(${Project_ID} PUBLIC tbb)
    endif()
endif()