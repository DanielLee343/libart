CC=gcc
CXX=g++
LD=g++
PREFIX=/usr/local
LIBDIR=$(PREFIX)/lib
INCLUDEDIR=$(PREFIX)/include
CFLAGS_BASE=-std=c99 -D_GNU_SOURCE -O3
CFLAGS ?= $(CFLAGS_BASE) $(CFLAGS_EXTRA)
CXXFLAGS=-std=c++17 -O3
LDFLAGS = -lmemkind -lnuma
SHCFLAGS=$(CFLAGS) $(LDFLAGS) -fPIC
SHCXXFLAGS=$(CXXFLAGS) -fPIC
SHLINKFLAGS=-shared

all:	src/libart.so

src/libart.so:	src/libart.o src/visit_node.o src/hot_cache.o src/common.o
	$(LD) $(SHLINKFLAGS) -o $@ $^

src/art.c:	src/art.h

src/libart.o:	src/art.c
	$(CC) $(SHCFLAGS) -o $@ -c $<

src/common.o:	src/common.c src/art.h
	$(CC) $(SHCFLAGS) -o $@ -c $<

src/visit_node.o:	src/visit_node.cpp src/art.h
	$(CXX) $(SHCXXFLAGS) -o $@ -c $<

src/hot_cache.o:	src/hot_cache.cpp src/art.h
	$(CXX) $(SHCXXFLAGS) -o $@ -c $<

install:	src/libart.so
	mkdir -p $(DESTDIR)$(LIBDIR)
	mkdir -p $(DESTDIR)$(INCLUDEDIR)
	cp src/libart.so $(DESTDIR)$(LIBDIR)/libart.so
	chmod 555 $(DESTDIR)$(LIBDIR)/libart.so
	cp src/art.h $(DESTDIR)$(INCLUDEDIR)/art.h
	chmod 444 $(DESTDIR)$(INCLUDEDIR)/art.h

clean:
	rm -rf src/libart.so src/libart.o src/visit_node.o src/hot_cache.o src/common.o