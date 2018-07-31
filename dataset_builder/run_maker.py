from dataset_builder.tfrecord_maker import KittiEigenSplit

srcpath = '/media/ian/iandata/kitti_raw_data'
test_dir = '/home/ian/workspace/CompareDeVo/compare-de-vo/data'

kitti = KittiEigenSplit(srcpath, test_dir)
