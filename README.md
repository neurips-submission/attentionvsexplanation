# attentionvsexplanation
## Explanation vs Attention: A Two Player Game to obtain Attention for VQA

Pytorch implementation of an "Explanation vs Attention: A Two Player Game to obtain Attention for VQA" .
### Training Step:

    1. Download VQA dataset from [VQA site](https://visualqa.org/) and Ms-COCO images from [Microsoft site] (http://cocodataset.org/#download).
    
    [MDN_VQG Code](https://github.com/badripatro/Visual_Question_Generation)
    
    2. Create Vocab using: 
    
              --- python preprocess-vocab.py


    3. Preprocess the MSCOCO image file using : 
    
          --- python preprocessing/preprocess-images_vgg16.py
    
    4. To train model:
    
          --- ./train.sh    
          
    5. To evaluate model:
    
            --- ./evaluate.sh    
