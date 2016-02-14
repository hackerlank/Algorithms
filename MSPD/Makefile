all: solib test

solib:
	$(CXX) -O3 -c -fPIC -Wall -W mspd.cpp -I ../../ulib/include/ -L ../../ulib/lib/ -lgsl -lcblas -lglpk -lulib
	$(AR) csr libmspd.a mspd.o

test:
	$(CXX) -O3 -o test -fPIC -Wall -W test.cpp -L. -I ../../ulib/include/ -L ../../ulib/lib/ -lmspd -lgsl -lglpk -lcblas -lulib

clean:
	rm -rf *~ test libmspd.so *.o libmspd.a

.PHONY: solib test clean
