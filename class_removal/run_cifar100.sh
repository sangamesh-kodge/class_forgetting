for arch in vgg11_bn 
do
    for i in {0..9}
    do 
        CUDA_VISIBLE_DEVICES=0 python ./class_removal/main.py --dataset cifar100 --unlearn-class $(( 10*i )) --arch $arch \
         --forget-samples 990 --retain-samples 10 --max-batch-size 256 --test-batch-size 512 \
         --group-name 'final' --project-name "final" \
         --mode "baseline,sap" --mode-forget "baseline,sap" --projection-location "pre" --projection-type "baseline,I-(Mf-Mi)" \
         --scale-coff "100,300,1000" --scale-coff-forget "3,10,30,100"  
    done
done