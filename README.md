Here is a structured `README.md` file based on the details you provided:

---

# VoxelNet Implementation

This repository provides an implementation of **VoxelNet**: End-to-End Learning for Point Cloud Based 3D Object Detection using TensorFlow 2.0.0. This project is inspired by the work of Qiangui Huang and Xiaojian Ma. Thanks to them for their significant contributions, which have facilitated the reconstruction of this architecture and deepened the understanding of the non-explicit parts of the original paper.

## Dependencies

- Python 3.6
- TensorFlow 2.0.0
- OpenCV
- Numba

## Installation

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Compile the Cython module:**
   ```bash
   python3 setup.py build_ext --inplace
   ```

3. **Compile the evaluation code (Optional):**
   This step compiles the `Kitti_eval` project. Note that this is not required during training but can be used for evaluation after training.
   ```bash
   cd kitti_eval
   g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
   ```

4. **Grant execution permission to the evaluation script:**
   ```bash
   cd kitti_eval
   chmod +x launch_test.sh
   ```

## Data Preparation

1. Download the 3D KITTI detection dataset from [here](<dataset-url>). The following files are required:
   - Velodyne point clouds (29 GB)
   - Training labels of object data set (5 MB)
   - Camera calibration matrices of object data set (16 MB)
   - Left color images of object data set (12 GB)

2. Prepare cropped point cloud data:
   - Update the directories in `data/crop.py` and run the script to generate cropped data. Note that this will overwrite raw point cloud data.

3. Split the dataset into training and validation sets according to the [protocol](<protocol-url>), and rearrange the folders to have the following structure:
   ```
   └── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
       └── validation  <--- evaluation data
           ├── image_2
           ├── label_2
           └── velodyne
   ```

## Training

Run the training script with the following command:
```bash
python train.py \
--strategy="all" \
--n_epochs=160 \
--batch_size=2 \
--learning_rate=0.001 \
--small_addon_for_BCE=1e-6 \
--max_gradient_norm=5 \
--alpha_bce=1.5 \
--beta_bce=1 \
--huber_delta=3 \
--dump_vis="no" \
--data_root_dir="../DATA_DIR/T_DATA" \
--model_dir="model" \
--model_name="model6" \
--dump_test_interval=40 \
--summary_interval=10 \
--summary_val_interval=10 \
--summary_flush_interval=20 \
--ckpt_max_keep=10
```

## Evaluation

Run the prediction script with:
```bash
python predict.py \
--strategy="all" \
--batch_size=2 \
--dump_vis="yes" \
--data_root_dir="../DATA_DIR/T_DATA/" \
--dataset_to_test="validation" \
--model_dir="model" \
--model_name="model6" \
--ckpt_name=""
```

To compute model performance, run the `kitti_eval` project:
```bash
./kitti_eval/evaluate_object_3d_offline [DATA_DIR]/validation/label_2 ./predictions [output file]
```


