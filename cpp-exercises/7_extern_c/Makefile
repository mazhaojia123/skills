#------------------------
# compliers and flags
#------------------------
CC=clang
CPP=clang++

CXXFLAGS= -fPIC -std=c++17
CFLAGS=-fPIC
COMMON= -Iinclude/

AR=ar
ARFLAGS=rcs
# OPTS=-Ofast
# LDFLAGS= -lm -pthread 


#------------------------
# names of srcs and objs
#------------------------
SRCDIR=./src/
OBJDIR=./build/

OBJ_NAME = f.o
EXECOBJ_NAME=main.o
EXEC_NAME=main

ALIB= $(OBJDIR)libec.a

#------------------------
# roles
#------------------------
OBJS = $(addprefix $(OBJDIR), $(OBJ_NAME))
EXECOBJS=$(addprefix $(OBJDIR), $(EXECOBJ_NAME))
EXECS=$(addprefix $(OBJDIR), $(EXEC_NAME))


all: obj $(EXECS)

# EXECOBJ 与 .a 链接
$(EXECS): $(EXECOBJS) $(ALIB)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

# 将所有的 OBJ 编译成 .a
$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

# 也只有这个时候适合用%
# 将所有的源文件编译成.o，不论是 EXECOBJ 还是 OBJ
# $(OBJDIR)%.o: $(SRCDIR)%.cpp $(DEPS)
$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CPP) $(COMMON) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: $(SRCDIR)%.c
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj: 
	mkdir -p $(OBJDIR)

clean: 
	rm -rf $(OBJDIR) $(EXEC_NAME)

run: 
	./build/main