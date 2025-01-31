# Параллельное решение уравнений в частных производных с использованием метода конечных элементов
## Описание
Уравнения в частных производных (УЧП) играют ключевую роль в моделировании различных физических процессов, таких как теплопередача, динамика жидкостей, механика деформируемого твердого тела, электромагнитное поле и многие другие. Эти уравнения позволяют описывать и предсказывать поведение сложных систем с высокой степенью точности. Однако аналитическое решение УЧП возможно лишь для ограниченного числа задач, и в большинстве случаев требуется численное решение, что зачастую связано с большими затратами вычислительных ресурсов и времени.

Современные методы численного решения УЧП, такие как метод конечных разностей, метод конечных элементов и спектральные методы, позволяют эффективно приближать решения для сложных геометрий и условий. Однако, несмотря на достижения в области численного анализа и увеличения мощности современных компьютеров, решение крупномасштабных задач может занимать значительное время. Это становится особенно критичным при необходимости проведения множества вычислительных экспериментов, например, в оптимизационных задачах или при моделировании сложных физических явлений в реальном времени.

В последние годы всё большее внимание уделяется методам параллельных вычислений, которые позволяют существенно ускорить процесс решения численных задач за счёт распределения вычислительной нагрузки между несколькими процессорами. Использование параллелизма в численных методах для УЧП открывает новые возможности для решения крупных задач, которые ранее были недоступны из-за ограничений вычислительных мощностей.

Цель данной курсовой работы заключается в исследовании методов ускорения численного решения уравнений в частных производных с использованием параллельных вычислений. В работе будут рассмотрены основные подходы к параллелизации численных методов, а также проведен анализ их эффективности на различных примерах физических процессов. Особое внимание будет уделено вопросам балансировки нагрузки и оптимизации использования вычислительных ресурсов, что позволит добиться максимальной производительности при решении УЧП.

Таким образом, данная работа направлена на углубление понимания процессов параллельного вычисления в контексте численного решения УЧП и на разработку практических рекомендаций по эффективному использованию параллелизма для ускорения вычислительных процессов в различных областях физики и инженерии.
 Цель работы: исследовать возможности применения алгоритмов параллельного программирования для решения уравнений в частных производных


## Требования
* Операционная система: Linux 22.04
* Компилятор: GCC (GNU Compiler Collection)
* Пакетный менеджер: `apt`
* CMake версии 3.12 или выше

## Установка необходимых зависимостей
Обновите систему и установите основные инструменты для разработки:
```
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake git
```
Установите дополнительные зависимости, необходимые для DEAL.II:
```
sudo apt install libboost-all-dev libgsl-dev libtbb-dev libmuparser-dev libpetsc-dev libslepc-dev libp4est-dev
```
Установите OpenMPI:
```
sudo apt install libopenmpi-dev openmpi-bin
```
Проверьте установку MPI:
```
mpicc --version
mpirun --version
```
## Запуск
Компиляция:
```
bash recompile.sh
```
Выполнение
```
mpiexec -np <количество процессов> ./build/step-1
```

