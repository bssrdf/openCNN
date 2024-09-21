ARCH = 86 # 75 # modify this. Ampere=86
NAME = wgrad
NAME1 = ggrad
OUT = OPTSTS64
#MODE = PROF
#LBR = OPENCNN

all:
	nvcc src/openCNN_winograd.cu -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME) -D$(OUT)
	nvcc src/openCNN_winograd_ggml.cu -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME1) -D$(OUT)
	# nvcc src/openCNN_winograd.cu -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH)-o $(NAME) -D$(OUT)

clean:
	rm $(NAME)
