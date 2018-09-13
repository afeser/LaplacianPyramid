CC           = nvcc
BIN_DIR      = "bin"
SRC_DIR      = "src"
LIBRARY_PATH = "/home/afeser/Dropbox/Documents/Canavarr/Pratik/MyLibrary/lib"
INCLUDE_PATH = "/home/afeser/Dropbox/Documents/Canavarr/Pratik/MyLibrary/include"

LaplacianPyramid: createDirs
	$(CC) -g -o $(BIN_DIR)/LaplacianPyramid.bin -I$(INCLUDE_PATH) $(SRC_DIR)/LaplacianPyramid.cu -lcudart -lcuda -lppm -L$(LIBRARY_PATH) #-L/usr/local/cuda-9.2/lib64/ #kutuphaneyi de nvcc ile yaptigimda -lcudart ve -lcuda olmadiginda da hata verdi




createDirs:
	mkdir -p $(BIN_DIR)
