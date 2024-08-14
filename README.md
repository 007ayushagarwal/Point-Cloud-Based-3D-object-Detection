Voxelnet (tensorflow 2.0.0)
Image of Voxelnet Architecture

Implementation of VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection in tensorflow 2.0.0.
This project is based on the work of Qiangui Huang, (project) and Xiaojian Ma. Thanks to them for their tremendous job that allowed me to rebuild this architecture and understand the non explicit parts of the paper.


Dependencies
Python 3.6
Tensorflow 2.0.0
opencv
numba
Installation
Clone this repository
Compile the Cython module
$ python3 setup build_ext --inplace
Compile the evaluation code (Optional) This will compile the Kitti_eval project which I decided not to run during the training (during the dump tests). But at the end, of the training, you may run this evaluation code with the model of your choice. In the training script, it is commented. You may un-comment to run the kitti_eval during the training.
$ cd kitti_eval
$ g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
grant the execution permission to evaluation script
$ cd kitti_eval
$ chmod +x launch_test.sh
Data preparation
Download the 3D KITTI detection dataset from here. Data to download include:

Velodyne point clouds (29 GB): input data to VoxelNet
Training labels of object data set (5 MB): input label to VoxelNet
Camera calibration matrices of object data set (16 MB): for visualization of predictions
Left color images of object data set (12 GB): for visualization of predictions
In this project, we use the cropped point cloud data for training and validation. Point clouds outside the image coordinates are removed. Update the directories in data/crop.py and run data/crop.py to generate cropped data. Note that cropped point cloud data will overwrite raw point cloud data.

Split the training set into training and validation set according to the protocol here. And rearrange the folders to have the following structure:

└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
       └── validation  <--- evaluation data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
Train
Run train.py. You can find the meaning of each hyperparameter in the script file.

$ !python train.py \
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
--ckpt_max_keep=10 \
Evaluate
Run predict.py.
!python predict.py \
--strategy="all" \
--batch_size=2 \
--dump_vis="yes" \
--data_root_dir="../DATA_DIR/T_DATA/" \
--dataset_to_test="validation" \
--model_dir="model" \
--model_name="model6" \
--ckpt_name="" \
Then, run the kitty_eval project to compute the performances of the model.
./kitti_eval/evaluate_object_3d_offline [DATA_DIR]/validation/label_2 ./predictions [output file]
What's new
Tensorflow 2.0.0
Data pipeline with tensorflow dataset api
Eager mode (with use of autograph for speed)
Use of a variant of the smooth-l1 loss for the regression loss (use of Huber loss)
Use of tf.distribute for the multi gpu training (still in process, only one gpu training works for now)
Non use of the first type of data augmentation (may lead in a decrease of performance)
Performances
(ongoing) I've just finished the project, and start training it. But before that, I did a lot of tests to challenge the archictecture. One of them is overfitting the model on a small training set in a few steps in order to check if i built a model able to learn anything at all, results, below.(PS : I tried to be faithful as much as I could to the paper).

perf perf2

Coming next
Train of models for car detection
Finish multi gpu interface
Train models for Pedestrian and Cyclist detection
Try new features that I'll communicate soon
