CC = g++

LDIR = /home/kgriffin/.conda-envs/AC-AMAR/lib
IDIR = /home/kgriffin/.conda-envs/AC-AMAR/include/python3.12

LIBS = -lpython3.12

main: main.cpp
	$(CC) main.cpp func_4d.cpp -o output -L $(LDIR) $(LIBS) -I $(IDIR)

