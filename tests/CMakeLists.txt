include(GoogleTest)

# Unit test executable
file(GLOB_RECURSE TEST_SOURCES unit/*.cpp)

add_executable(anacapa_tests ${TEST_SOURCES})

target_link_libraries(anacapa_tests PRIVATE
    anacapa_lib
    GTest::gtest_main
)

target_include_directories(anacapa_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

gtest_discover_tests(anacapa_tests)
