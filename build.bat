nvcc src/openCNN_winograd.cu -lcudnn -m64 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\lib\x64" -arch=compute_86 -lineinfo -code=sm_86 -o wgrad -DOPTSTS64
nvcc src/openCNN_winograd_ggml.cu -lcudnn -m64 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\lib\x64" -arch=compute_86 -lineinfo -code=sm_86 -o ggrad -DOPTSTS64
nvcc src/openCNN_winograd_1x16x8.cu -lcudnn -m64 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\lib\x64" -arch=compute_86 -lineinfo -code=sm_86 -o ggrad1x16x8 -DOPTSTS64
nvcc src/openCNN_winograd_1x8x64.cu -lcudnn -m64 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\lib\x64" -arch=compute_86 -lineinfo -code=sm_86 -o ggrad1x8x64 -DOPTSTS64
nvcc src/openCNN_winograd_32Tx64x8.cu -lcudnn -m64 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.9.7.29_cuda12\lib\x64" -arch=compute_86 -lineinfo -code=sm_86 -o ggrad32Tx64x8 -DOPTSTS64