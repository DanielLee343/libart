CC = gcc
CFLAGS = -O2
LDFLAGS = -lart -lmemkind -lnuma -lpthread
TARGET = no_op
SRC = no_op.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)