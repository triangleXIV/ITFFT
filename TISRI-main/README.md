## Data 
- Twitter datasets for our task: the processed pkl files are in floder  `./data/Sentiment_Analysis/twitter201x/` . The original tweets, images and sentiment annotations can be download from [https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)

## Image Processing 
Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth), and put the pre-trained ResNet-152 model under the folder './model/resnet/"

## Code Usage
Note that you should change data path.
i is 'twitter2015' or 'twitter2017'.

- Training
```
python train.py 
    --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --imagefeat_dir ./data/twitter_images/ \
    --VG_imagefeat_dir ./data/twitter_images/ \
    --output_dir ./log/ \
	--recon_loss_ratio 0.8
```

- Inference
```
python test.py 
    --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --imagefeat_dir ./data/twitter_images/ \
    --VG_imagefeat_dir ./data/twitter_images/ \
    --output_dir ./log/ \
	--recon_loss_ratio 0.8 \
    --model_file pytorch_model.bin
```
