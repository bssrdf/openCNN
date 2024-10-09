ARCH = 86 # 75 # modify this. Ampere=86
NAME = wgrad
NAME1 = ggrad
NAME2 = ggrad32Tx64x8
NAME3 = ggrad1x8x64
OUT = OPTSTS64
#MODE = PROF
#LBR = OPENCNN

all:
	# nvcc src/openCNN_winograd.cu -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME) -D$(OUT)
	# nvcc src/openCNN_winograd_ggml.cu -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME1) -D$(OUT)
	nvcc src/openCNN_winograd_32Tx64x8.cu -Xptxas="-v" -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME2) -D$(OUT)
	# nvcc src/openCNN_winograd_1x8x64.cu -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME3) -D$(OUT)
	# nvcc src/openCNN_winograd.cu -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH)-o $(NAME) -D$(OUT)

clean:
	rm $(NAME)
