CC           = nvcc
BIN_DIR      = "bin"
SRC_DIR      = "src"
LIBRARY_PATH = "/home/afeser/Dropbox/Documents/Canavarr/Pratik/MyLibrary/lib"
INCLUDE_PATH = "/home/afeser/Dropbox/Documents/Canavarr/Pratik/MyLibrary/include"

createDirs:
	mkdir -p $(BIN_DIR)

laplacianpyramid: createDirs
	$(CC) -g -o $(BIN_DIR)/LaplacianPyramidV2.bin -I$(INCLUDE_PATH) $(SRC_DIR)/LaplacianPyramidV2.cu -lcudart -lcuda -lppm -L$(LIBRARY_PATH) -L/usr/local/cuda-9.2/lib64/ #kutuphaneyi de nvcc ile yaptigimda -lcudart ve -lcuda olmadiginda da hata verdi
gaussianpyramid:
	$(CC) -g -o $(BIN_DIR)/GaussianPyramid.bin -I$(INCLUDE_PATH) $(SRC_DIR)/GaussianPyramid.cu -lppm -L$(LIBRARY_PATH)
gaussianfilter5x5:
	$(CC) -g -o $(BIN_DIR)/GaussianFilter5x5GPU.bin -I$(INCLUDE_PATH) $(SRC_DIR)/GaussianFilter5x5GPU.cu -lppm -L$(LIBRARY_PATH)
gaussianfilter:
	$(CC) -g -o $(BIN_DIR)/GaussianFilterGPU.bin -I$(INCLUDE_PATH) $(SRC_DIR)/GaussianFilterGPU.cu -lppm -L$(LIBRARY_PATH)
sharpenfiltergpu:
	$(CC) -g -o $(BIN_DIR)/SharpenFilterGPU.bin -I$(INCLUDE_PATH) $(SRC_DIR)/SharpenFilterGPU.cu -lppm -L$(LIBRARY_PATH)
howthreadworks:
	$(CC) -g -o $(BIN_DIR)/HowThreadWorks.bin -I$(INCLUDE_PATH) $(SRC_DIR)/HowThreadWorks.cu -lppm -L$(LIBRARY_PATH)
stencilkernel:
	$(CC) -g -o $(BIN_DIR)/StencilKernel.bin -I$(INCLUDE_PATH) $(SRC_DIR)/StencilKernel.cu -lppm -L$(LIBRARY_PATH)
testkernel:
	$(CC) -g -o $(BIN_DIR)/TestKernel.bin $(SRC_DIR)/TestKernel.cu
simpleFilter:
	gcc -g -o $(BIN_DIR)/SimpleFilter.bin -I$(INCLUDE_PATH) $(SRC_DIR)/SimpleFilterCPU.cpp -lppm -L$(LIBRARY_PATH) ### COK GARIP!!!! SimpleFilter.cpp oldugunda hata veriyor SimpleFilter.c oldugunda duzgun calisiyor...

build:
	$(CC) -g -o $(BIN_DIR)/HelloWorldGPU.bin $(SRC_DIR)/HelloWorldGPU.cu
	$(CC) -g -o $(BIN_DIR)/HelloWorldCPU.bin $(SRC_DIR)/HelloWorldCPU.cu
	$(CC) -g -o $(BIN_DIR)/GPUAdder.bin      $(SRC_DIR)/GPUAdder.cu
	$(CC) -g -o $(BIN_DIR)/GPUAdderCorrect.bin      $(SRC_DIR)/GPUAdderCorrect.cu
