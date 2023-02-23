# If you want to use Intel compilers, you need to modify the following
# flags to link with MKL. 

CC = gcc
CPP = g++
OPT = -O3
CFLAGS = -Wall  $(OPT)  -m64 -I/opt/OpenBLAS/include -I/usr/local/include/vcl2 -mavx2 -mfma -march=core-avx2 -fopt-info-vec-optimized -ftree-loop-optimize
LDFLAGS = -Wall
# librt is needed for clock_gettime
# LDLIBS = -lrt    -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl 
LDLIBS = -lrt -L/opt/OpenBLAS/lib -lopenblas

targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o  

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CPP) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

%.o : %.cpp
	$(CPP) -std=c++17 -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
