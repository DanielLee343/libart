CC=gcc
LD=gcc
PREFIX=/usr/local
LIBDIR=$(PREFIX)/lib
INCLUDEDIR=$(PREFIX)/include
CFLAGS_BASE=-g -std=c99 -D_GNU_SOURCE -Wall -Werror -O3
CFLAGS ?= $(CFLAGS_BASE) $(CFLAGS_EXTRA)
LDFLAGS = -lmemkind -lnuma
SHCFLAGS=$(CFLAGS) $(LDFLAGS) -fPIC
SHLINKFLAGS=-shared

all:	src/libart.so

src/libart.so:	src/libart.o
	$(LD) $(SHLINKFLAGS) -o $@ $<

src/art.c:	src/art.h

src/libart.o:	src/art.c
	$(CC) $(SHCFLAGS) -o $@ -c $<

install:	src/libart.so
	mkdir -p $(DESTDIR)$(LIBDIR)
	mkdir -p $(DESTDIR)$(INCLUDEDIR)
	cp src/libart.so $(DESTDIR)$(LIBDIR)/libart.so
	chmod 555 $(DESTDIR)$(LIBDIR)/libart.so
	cp src/art.h $(DESTDIR)$(INCLUDEDIR)/art.h
	chmod 444 $(DESTDIR)$(INCLUDEDIR)/art.h

clean:
	rm -rf src/libart.so src/libart.o