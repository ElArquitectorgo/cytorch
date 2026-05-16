CC=gcc
CFLAGS=-Wall -Wextra -O2 -Iinclude

SRC=$(wildcard src/*.c)
MAIN=$(filter-out src/main.c, $(SRC))
TESTS=tests/test_autograd.c

all:
	$(CC) $(CFLAGS) $(SRC) -o cytorch
	./cytorch

test:
	$(CC) $(CFLAGS) $(MAIN) $(TESTS) -o test
	./test

clean:
	rm -f src/*.o tests/*.o cytorch test