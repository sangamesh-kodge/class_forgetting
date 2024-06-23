for arch in vgg11_bn 
do
    unlearn_class=3
    for missing_class in 0 1 2 3 4 5 6 7 8 9 
    do  
        # Our
        CUDA_VISIBLE_DEVICES=0 python ./extra_code/missing_class.py --dataset cifar10 --data-path ../data \
            --unlearn-class $unlearn_class --ignore-class $missing_class --arch $arch \
            --group-name 'final' --project-name "tmlr-submission-missingclass" \
            --seed 42 \
            --unlearn-method "our" --our-samples 999 --our-alpha-r 100  --our-alpha-f 10  
    done
done