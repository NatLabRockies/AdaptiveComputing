CC = g++

LDIR = ~/.conda-envs/AC-AMAR-3.11-v2/lib
IDIR = ~/.conda-envs/AC-AMAR-3.11-v2/include/python3.11

LIBS = -lpython3.11

main: main.cpp
	$(CC) main.cpp func_4d.cpp -o output -L $(LDIR) $(LIBS) -I $(IDIR)

