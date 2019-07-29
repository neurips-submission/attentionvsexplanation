# attentionvsexplanation
## Explanation vs Attention: A Two Player Game to obtain Attention for VQA

Pytorch implementation of an "Explanation vs Attention: A Two Player Game to obtain Attention for VQA" .
### Training Step:

    1. Download VQA dataset from VQA site[https://visualqa.org/] and Ms-COCO images from Microsoft[ http://cocodataset.org/#download] site.
    2. Create train,val and test json file.
    3. Preprocess the MSCOCO image file using : 
    
    ``` python preprocessing/preprocess-images_vgg16.py```
    
    4. Create Vocab using: 
    
    ```python preprocess-vocab.py``` 
    
    5. Train using : ./train.sh
