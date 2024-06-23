for arch in vgg11_bn 
do
    for i in {0..9}
    do  
        # Our
        CUDA_VISIBLE_DEVICES=0 python ./extra_code/tune_our.py --dataset cifar10 --data-path ../data \
            --unlearn-class $i --arch $arch \
            --group-name 'final' --project-name "tmlr-submission-hptune" \
            --seed 42 \
            --unlearn-method "our" --our-samples 900 --our-alpha-r "10,30,100,300,1000"  --our-alpha-f "1,3,10,30,100" 
    done
done