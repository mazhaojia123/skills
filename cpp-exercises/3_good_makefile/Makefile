CC=clang
CPP=clang++
AR=ar
ARFLAGS=rcs
# OPTS=-Ofast
# LDFLAGS= -lm -pthread 
COMMON= -Iinclude/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

SRCDIR=./src/
DEPS = $(wildcard include/*.h)

OBJDIR=./build/
OBJ = a.o b.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

ALIB= $(OBJDIR)libgood.a

EXEC=main
EXECOBJ=main.o
EXECOBJS=$(addprefix $(OBJDIR), $(EXECOBJ))

all: obj $(EXEC)

# EXECOBJ 与 .a 链接
$(EXEC): $(EXECOBJS) $(ALIB)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

# 将所有的 OBJ 编译成 .a
$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

# 也只有这个时候适合用%
# 将所有的源文件编译成.o，不论是 EXECOBJ 还是 OBJ
# $(OBJDIR)%.o: $(SRCDIR)%.cpp $(DEPS)
$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

obj: 
	mkdir -p $(OBJDIR)

clean: 
	rm -rf $(OBJDIR) $(EXEC)