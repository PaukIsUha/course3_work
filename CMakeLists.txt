cmake_minimum_required(VERSION 3.1)
project(DealIITutorial)

# Укажите минимальную версию C++.
set(CMAKE_CXX_STANDARD 17)

# Найдите пакет deal.II. Обычно он правильно устанавливает свои пути и переменные.
find_package(deal.II REQUIRED)
include_directories(${DEAL_II_INCLUDE_DIRS})
link_directories(${DEAL_II_LIBRARY_DIRS})
add_definitions(${DEAL_II_DEFINITIONS})

# Найдите MPI
find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH})

# Добавьте исполняемый файл вашей программы.
add_executable(step-1 main.cpp)
target_link_libraries(step-1 ${DEAL_II_LIBRARIES} ${MPI_C_LIBRARIES})
