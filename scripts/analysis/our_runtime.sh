for arch in vgg11_bn 
do
    for i in {0..9}
    do  
        # Our
        CUDA_VISIBLE_DEVICES=0 python ./extra_code/runtime.py --dataset cifar10 --data-path ../data \
            --unlearn-class $i --arch $arch \
            --group-name 'final' --project-name "tmlr-submission-runtime" \
            --seed 42 \
            --unlearn-method "our" --our-samples 900 --our-alpha-r 100  --our-alpha-f 3  
    done
done