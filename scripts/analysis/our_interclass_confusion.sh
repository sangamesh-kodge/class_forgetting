for arch in vit_b_16 
do
    for i in 2 
    do  
        # Our
        CUDA_VISIBLE_DEVICES=0 python ./extra_code/interclass_confusion.py --dataset imagenet --data-path <data-path> \
            --unlearn-class $(( i*100 )) --arch $arch \
            --group-name 'final' --project-name "tmlr-submission-confusion" \
            --seed 42 \
            --unlearn-method "our" --our-samples 999 --our-alpha-r 100  --our-alpha-f 10  
    done
done