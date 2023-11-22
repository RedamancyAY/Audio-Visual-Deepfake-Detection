# LCL-AVD: A Lightweight Collaborative Learning Network for Joint Audio-Visual Deepfake Detection

This is the implemtation code of python for our paper: `LCL-AVD: A Lightweight Collaborative Learning Network for Joint Audio-Visual Deepfake Detection`. We first illustrate the data preparation and then show the train/test code.



## Data Preparation

### Dataset Downloading & Uncompression


#### DF-TIMIT. 

It use Faceswap-GAN on the Vid-TIMIT dataset, please downlaod both the [Vid-TIMIT]( http://conradsanderson.id.au/vidtimit/) and [DF-TIMIT](https://zenodo.org/record/4068245#.ZFsDunZBy4Q) datasets. Uncompress them, rename Vid-TIMIT folder into `real`, and move the `real` folder into the DF-TIMIT. The structure of the DF-TIMIT folder is: 
```json
- DeepfakeTIMIT
    - higher_quality
        - fadgo
            - sa1-video-fram1.avi
            - sa1-video-fram1.wav
            - ...
        - faks0
        - ...
    - lower_quality
        - fadgo
        - faks0
        - ...
    - real
        - fadgo
        - faks0
        - ...
    - README.txt
```
Note that you need combine all the video frames into `.avi` videos and put `.avi` video and `.wav` audio together.

#### FakeAVCeleb. 

Since we use VoxCeleb2 as the complement for the real videos, you need to download both the [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) and [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) datasets. The used 2000 videos of VoxCeleb2 is showed in  `data/info/VoxCeleb2.txt`. Please create a folder `...../FakeAVCeleb_v1.2/VoxCeleb2` and put these videos in it. Then, the structure of the FakeAVCeleb folder is: 
```json
FakeAVCeleb_v1.2
├── RealVideo-RealAudio
├── FakeVideo-RealAudio
├── RealVideo-FakeAudio
├── FakeVideo-FakeAudio
├── VoxCeleb2
├   └──xxx.mp4
├── README.txt
└── meta_data.csv
```



#### DFDC

Download the [DFDC](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) and extract the videos. Then, the structure of the DFDC folder is: 

```json
dfdc
├── dfdc_train_part_0
├──────xxx.mp4
├── dfdc_train_part_1
├── dfdc_train_part_xxx
└── dfdc_train_part_49
```
We use 18000 videos of DFDC to evaluate all the models. The list of the 18000 videos is listed in `data/splits/DFDC.txt`. Besides,
* please move and rename the 'data/info/DFDC_metadata.csv' into the uncompressed folder: '...../dfdc/metadata.csv'.


### Dataset Preprocessing

After dataset downloading and uncompression, we can obtain the `root_path` for each dataset:
```text
...../DeepfakeTIMIT
...../FakeAVCeleb_v1.2
...../dfdc
```

Now, we need to create `data_path` folders for each dataset to contain the preprocessing results.
```text
...../DF-TIMIT
...../FakeAVCeleb+
...../DFDC
```




To preprocess the datasets, you need to cd the `data` folder in terminal, and use the following cmd:
```bash
python dataset_preprocess.py --dataset DF-TIMIT --root_path "...../DeepfakeTIMIT"    --data_path "...../DF-TIMIT"
python dataset_preprocess.py --dataset DF-TIMIT --root_path "...../FakeAVCeleb_v1.2" --data_path "...../FakeAVCeleb+"
python dataset_preprocess.py --dataset DFDC     --root_path "...../dcdc"             --data_path "...../DFDC"
```
where `root_path` and `data_path` are paths of above created folders for each datasets. Above command will extract all the frames and the audios for the videos of each dataset. You should change them for your custom folder in your computer. 

For DFDC dataset, since we use S3FD to extract the face, you need to use it (or other face extractors) to extract the face and replace the figures in "...../DFDC/s3fd".


Finally, please update the `root_path` and `data_path` for each dataset in `config/dataset.py`.


## Evaluation

You need to specify the logger path (save model checkpoint & logger) in the train.py (line 137, 146, 163, 179)

### Train

Train the small version of our LCL-AVD:
```
python train.py --gpu 0 --cfg 'LCL-AVD-s/DF_TIMIT_HQ'  --earlystop 20
python train.py --gpu 1 --cfg 'LCL-AVD-s/DF_TIMIT_LQ' --earlystop 20
python train.py --gpu 1  --cfg 'LCL-AVD-s/FakeAVCeleb/raw' 
python train.py --gpu 0  --cfg 'LCL-AVD-s/FakeAVCeleb/raw-fsgan' 
python train.py --gpu 1  --cfg 'LCL-AVD-s/FakeAVCeleb/raw-faceswap' 
python train.py --gpu 0  --cfg 'LCL-AVD-s/DFDC'
```

Train the large version of our LCL-AVD:
```
python train.py --gpu 0 --cfg 'LCL-AVD-l/DF_TIMIT_HQ'  --earlystop 20
python train.py --gpu 1 --cfg 'LCL-AVD-l/DF_TIMIT_LQ' --earlystop 20
python train.py --gpu 1  --cfg 'LCL-AVD-l/FakeAVCeleb/raw'
python train.py --gpu 0  --cfg 'LCL-AVD-l/FakeAVCeleb/raw-fsgan'
python train.py --gpu 1  --cfg 'LCL-AVD-l/FakeAVCeleb/raw-faceswap'
python train.py --gpu 0  --cfg 'LCL-AVD-l/DFDC';
```
After training finished, the program will test the best model (best auc on val dataset) on the test set.

### Test 
To test the models, you can add `--test 1 --test_version xxx` to above cmd.