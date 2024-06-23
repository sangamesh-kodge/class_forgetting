
# Download torch weight for 
# vgg11_bn(https://download.pytorch.org/models/vgg11_bn-6002323d.pth)
# vit_b_16 (https://download.pytorch.org/models/vit_b_16-c867db91.pth)
# store these weights in location ->class_removal/pretrained_models/imagenet_<arch>.pt 

for arch in vgg11_bn vit_b_16 
do
    for i in {0..9}
    do  
        # Our
        CUDA_VISIBLE_DEVICES=0 python ./main.py --dataset imagenet --data-path <data-path> \
            --unlearn-class $(( i*100 )) --arch $arch \
            --group-name 'final' --project-name "tmlr-submission" \
            --seed 42 \
            --unlearn-method "our" --our-samples 999 --our-alpha-r 100  --our-alpha-f 10 \
            --do-mia
    done
done
