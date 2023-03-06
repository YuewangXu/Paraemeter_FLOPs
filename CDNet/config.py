from os.path import join

test_thread_num = 8  # Thread-num used for the test
device = '5'  # GPU index
batch_size = 10  # Batch-size used for the test

checkpoint_path = './CDNet_2.pth'  # The file path of pre-trained checkpoint, e.g., '/mnt/jwd/code/CDNet.pth'
img_base = '/root/sharedatas/YuewangXu/RGB_Depth_Codes/datasets/TriTransNet-Dataset/test/STERE/RGB'  # RGB-images directory path, e.g., '/mnt/jwd/data/images/'
depth_base = '/root/sharedatas/YuewangXu/RGB_Depth_Codes/datasets/TriTransNet-Dataset/test/STERE/depth'  # Depth-maps directory path, e.g., '/mnt/jwd/data/depths/'
save_base = './result/CDNet/STERE'  #  The directory path used to save the predictions, e.g., './Predictions/'

test_roots = {'img': img_base,
              'depth': depth_base}