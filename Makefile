CC=gcc
CFLAGS=-Wall -Wextra -O2 -Iinclude

SRC=$(wildcard src/*.c)
OBJ=$(SRC:.c=.o)

all: cytorch

cytorch: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o cytorch

clean:
	rm -f src/*.o cytorch