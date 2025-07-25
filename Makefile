CC=gcc
LD=gcc
PREFIX=/usr/local
LIBDIR=$(PREFIX)/lib
INCLUDEDIR=$(PREFIX)/include
CFLAGS=-std=c99 -D_GNU_SOURCE -Wall -Werror -O3
SHCFLAGS=$(CFLAGS) -fPIC
SHLINKFLAGS=-shared

SRC_DIR=src
OBJ_DIR=$(SRC_DIR)
INCLUDES=$(SRC_DIR)/art.h $(SRC_DIR)/node_allocator.h

all: $(SRC_DIR)/libart.so

$(SRC_DIR)/libart.so: $(OBJ_DIR)/libart.o $(OBJ_DIR)/node_allocator.o
	$(LD) $(SHLINKFLAGS) -o $@ $^

$(OBJ_DIR)/libart.o: $(SRC_DIR)/art.c $(INCLUDES)
	$(CC) $(SHCFLAGS) -o $@ -c $<

$(OBJ_DIR)/node_allocator.o: $(SRC_DIR)/node_allocator.c $(SRC_DIR)/node_allocator.h
	$(CC) $(SHCFLAGS) -o $@ -c $<

install: $(SRC_DIR)/libart.so
	mkdir -p $(DESTDIR)$(LIBDIR)
	mkdir -p $(DESTDIR)$(INCLUDEDIR)
	cp $(SRC_DIR)/libart.so $(DESTDIR)$(LIBDIR)/libart.so
	chmod 555 $(DESTDIR)$(LIBDIR)/libart.so
	cp $(SRC_DIR)/art.h $(DESTDIR)$(INCLUDEDIR)/art.h
	cp $(SRC_DIR)/node_allocator.h $(DESTDIR)$(INCLUDEDIR)/node_allocator.h
	chmod 444 $(DESTDIR)$(INCLUDEDIR)/art.h
	chmod 444 $(DESTDIR)$(INCLUDEDIR)/node_allocator.h

clean:
	rm -rf src/libart.so src/*.o