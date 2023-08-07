# Simple Image Retrieval

## Setup
### Install requirements
<!-- `pip install pre-commit==2.13.0` -->
`pip install -r requirements.txt`
* Dev only

`pre-commit install`

### Download dataset
* Download the dataset and extract it to `data/` directory.

## Usage
### Generate feature vector
```
usage: main.py generate [-h] --dataset-path DATASET_PATH --output-dir OUTPUT_DIR

options:
  -h, --help            show this help message and exit
  --dataset-path DATASET_PATH, -dp DATASET_PATH
                        Dataset path
  --output-dir OUTPUT_DIR, -od OUTPUT_DIR
                        Feature vector output dir
```

* Example
```
python main.py generate --dataset-path data/simple_image_retrieval_dataset/image-db --output-dir .
```

### Search similar image
```
usage: main.py search [-h] --input-image INPUT_IMAGE --dataset-path DATASET_PATH --vectors-path VECTORS_PATH

options:
  -h, --help            show this help message and exit
  --input-image INPUT_IMAGE, -im INPUT_IMAGE
                        Input image
  --dataset-path DATASET_PATH, -dp DATASET_PATH
                        Dataset path
  --vectors-path VECTORS_PATH, -vp VECTORS_PATH
                        Vectors file path
```

* Example
```
python main.py search \
    --input-image data/simple_image_retrieval_dataset/test-cases/car.jpg \
    --dataset-path data/simple_image_retrieval_dataset/image-db \
    --vectors-path ./vectorized_dataset.npy
```

### Run tests
```
pytest
```
