CC=gcc
CFLAGS=-Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm

SRC=$(wildcard src/*.c)
MAIN=$(filter-out src/main.c, $(SRC))
TESTS=tests/test_autograd.c

all:
	$(CC) $(CFLAGS) $(SRC) -o cytorch $(LDFLAGS)
	./cytorch

test:
	$(CC) $(CFLAGS) $(MAIN) $(TESTS) -o test $(LDFLAGS)
	./test

clean:
	rm -f src/*.o tests/*.o cytorch test