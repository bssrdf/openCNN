echo "in_n,in_c,in_h,filt_k,filt_w,openCNN_sec,openCNN_flops,cuDNN_sec,cuDNN_flops"

#../wgrad 32 64 224 224 64 64 3 3
# ../wgrad 64 64 224 224 64 64 3 3
# ../wgrad 64 10 224 224 64 10 3 3
# ../wgrad 32 8 21 21 64 8 3 3
../ggrad 1 320 24 24 640 320 3 3
# ../wgrad 256 8 160 160 64 8 3 3
# ../wgrad 64 5 224 224 64 5 3 3
#../wgrad 32 8 1024 1024 64 8 3 3
#../wgrad 32 64 56 56 64 64 3 3

# for n in 32 64 96 128;
#     do
#         ../wgrad $n 64 56 56 64 64 3 3
#     done

# for n in 32 64 96 128; 
#     do
#         ../wgrad $n 128 28 28 128 128 3 3
#     done

# for n in 32 64 96 128; 
#     do
#         ../wgrad $n 256 14 14 256 256 3 3
#     done
    
# for n in 32 64 96 128; 
#     do
#         ../wgrad $n 512 7 7 512 512 3 3
#     done
        
