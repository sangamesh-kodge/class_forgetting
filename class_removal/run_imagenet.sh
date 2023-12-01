
# Download torch weight for 
# vgg11_bn(https://download.pytorch.org/models/vgg11_bn-6002323d.pth)
# vit_b_16 (https://download.pytorch.org/models/vit_b_16-c867db91.pth)
# store these weights in location ->class_removal/pretrained_models/imagenet_<arch>.pt 
for arch in vgg11_bn vit_b_16
do
    for i in {0..9}
    do 
        CUDA_VISIBLE_DEVICES=0 python ./class_removal/main.py --dataset imagenet --data-path <data-path> --unlearn-class $(( 100*i )) --arch $arch \
         --forget-samples 500 --retain-samples 1 --max-batch-size 128 --test-batch-size 256 --val-set-mode \
         --group-name 'final' --project-name "final" \
         --mode "baseline,sap" --mode-forget "baseline,sap" --projection-location "pre" --projection-type "baseline,I-(Mf-Mi)" \
         --scale-coff "10,30,100,300,1000,3000" --scale-coff-forget "3,10,30,100,300"  
    done
done