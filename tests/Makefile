CC = gcc
CFLAGS = -O2
LDFLAGS = -lart -lmemkind -lnuma -lpthread
TARGET = value
SRC = value.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)