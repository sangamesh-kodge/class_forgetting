for arch in resnet18 
do
    for i in {0..9}
    do  
        # Our
        CUDA_VISIBLE_DEVICES=0 python ./main.py --dataset cifar100  --data-path ../data \
            --unlearn-class $(( i*10 )) --arch $arch \
            --group-name 'final' --project-name "tmlr-submission" \
            --seed 42 \
            --unlearn-method "our" --our-samples 990 --our-alpha-r 1000  --our-alpha-f 30 \
            --do-mia-ulira   --plot-mia-roc  
    done
done