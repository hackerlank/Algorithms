all: solib test

solib:
	$(CXX) -O3 -c -fPIC -Wall -W pald.cpp -I ../ulib/include/ -L ../ulib/lib/ -lgsl -lcblas -lglpk -lulib
	$(AR) csr libpald.a pald.o

test:
	$(CXX) -O3 -o test -fPIC -Wall -W test.cpp -L. -I ../ulib/include/ -L ../ulib/lib/ -lpald -lgsl -lglpk -lcblas -lulib

clean:
	rm -rf *~ test libpald.so *.o libpald.a

.PHONY: solib test clean
