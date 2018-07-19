import h5py
import random
import numpy as np
from scipy.misc import imread, imshow
from PIL import Image

class DataGenerator:
    """
    Class for a generator that reads in data from the HDF5 file, one batch at
    a time, converts it into the jigsaw, and then returns the data
    """

    def __init__(self, conf, maxHammingSet):
        """
        Explain
        """
        self.data_path = conf.data_path
        self.numChannels = conf.numChannels
        self.numCrops = conf.numCrops
        self.cropSize = conf.cropSize
        self.cellSize = conf.cellSize
        self.tileSize = conf.tileSize
        self.colorJitter = conf.colorJitter
        self.batchSize = conf.batchSize
        self.meanTensor, self.stdTensor = self.get_stats()
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        self.batch_counter()
        self.numClasses = self.maxHammingSet.shape[0]   # i.e. number of jigsaw types

    def get_stats(self):
        h5f = h5py.File(self.data_path, 'r')
        mean = h5f['train_mean'][:].astype(np.float32)
        std = h5f['train_std'][:].astype(np.float32)
        h5f.close()
        return mean, std

    def batch_counter(self):
        h5f = h5py.File(self.data_path, 'r')
        self.numTrainBatch = 1 + h5f['train_img'][:].shape[0] // self.batchSize
        self.numValBatch = 1 + h5f['val_img'][:].shape[0] // self.batchSize
        self.numTestBatch = 1 + h5f['test_img'][:].shape[0] // self.batchSize
        h5f.close()
        self.batchIndexTrain = 0
        self.batchIndexVal = 0
        self.batchIndexTest = 0

    def __data_generation_normalize(self, x1):
        """
        Explain
        """
        x1 -= self.meanTensor
        x1 /= self.stdTensor
        sz = min(self.batchSize,x1.shape[0])
        # This implementation modifies each image individually
        y = np.empty(sz)#self.batchSize)
        # Python list of 4D numpy tensors for each channel
        X = [np.empty((sz, self.tileSize, self.tileSize, self.numChannels), np.float32)
             for _ in range(self.numCrops)]
        # img1 = Image.fromarray(x1.astype(np.uint8)[0,:,:,:], 'RGB')
        # img.save('my.png')
        # img1.show()
        for image_num in range(sz):#self.batchSize):
            # Transform the image into its nine crops
            single_image, y[image_num] = self.create_croppings(x1[image_num])
            for image_location in range(self.numCrops):
                X[image_location][image_num, :, :, :] = single_image[:, :, :, image_location]
        return X, y

    def one_hot(self, y):
        """
        Explain
        """
        return np.array([[1 if y[i] == j else 0 for j in range(self.numClasses)] for i in range(y.shape[0])])

    def generate(self, mode='train'):
        """
        Explain
        """
        if mode == 'train':
            h5f = h5py.File(self.data_path, 'r')
            x = h5f['train_img']
            x = x[self.batchIndexTrain * self.batchSize:(self.batchIndexTrain + 1) * self.batchSize, ...]
            x_files = h5f['train_files'][self.batchIndexTrain * self.batchSize:(self.batchIndexTrain + 1) * self.batchSize]
            # img1 = Image.fromarray(x[0], 'RGB')
            # img.save('my.png')
            # img1.show()
            h5f.close()
            X, y = self.__data_generation_normalize(x.astype(np.float32))
            self.batchIndexTrain += 1  # Increment the batch index
            if self.batchIndexTrain == self.numTrainBatch:
                self.batchIndexTrain = 0
        elif mode == 'valid':
            h5f = h5py.File(self.data_path, 'r')
            x = h5f['val_img'][self.batchIndexVal * self.batchSize:(self.batchIndexVal + 1) * self.batchSize, ...]
            x_files = h5f['val_files'][self.batchIndexVal * self.batchSize:(self.batchIndexVal + 1) * self.batchSize]
            h5f.close()
            X, y = self.__data_generation_normalize(x.astype(np.float32))
            self.batchIndexVal += 1  # Increment the batch index
            if self.batchIndexVal == self.numValBatch:
                self.batchIndexVal = 0
        elif mode == 'test':
            h5f = h5py.File(self.data_path, 'r')
            x = h5f['test_img']
            x=x[self.batchIndexTest * self.batchSize:min((self.batchIndexTest + 1) * self.batchSize,x.shape[0]), ...]
            x_files = h5f['test_files']
            x_files=x_files[self.batchIndexTest * self.batchSize:min((self.batchIndexTest + 1) * self.batchSize,x_files.shape[0])]
            h5f.close()
            X, y = self.__data_generation_normalize(x.astype(np.float32))
            self.batchIndexTest += 1  # Increment the batch index
            if self.batchIndexTest == self.numTestBatch:
                self.batchIndexTest = 0
        return np.transpose(np.array(X), axes=[1, 2, 3, 4, 0]), self.one_hot(y), x_files

    def randomize(self):
        """ Randomizes the order of data samples"""
        h5f = h5py.File(self.data_path, 'a')
        train_img = h5f['train_img'][:].astype(np.float32)
        permutation = np.random.permutation(train_img.shape[0])
        train_img = train_img[permutation, :, :, :]
        del h5f['train_img']
        h5f.create_dataset('train_img', data=train_img)
        h5f.close()

    def create_croppings(self, image):
        """
        Take in a 3D numpy array (256x256x3) and a 4D numpy array containing 9 "jigsaw" puzzles.
        Dimensions of the output array is 64 (height) x 64 (width) x 3 (colour channels) x 9(each cropping)

        The 3x3 grid is numbered as follows:
        0    1    2
        3    4    5
        6    7    8
        """
        # Jitter the colour channel, this is IMP
        image = self.color_channel_jitter(image)

        y_dim, x_dim = image.shape[:2]
        # Have the x & y coordinate of the crop
        crop_x = random.randrange(x_dim - self.cropSize)
        crop_y = random.randrange(y_dim - self.cropSize)
        # Select which image ordering we'll use from the maximum hamming set
        perm_index = random.randrange(self.numClasses)
        final_crops = np.zeros((self.tileSize, self.tileSize, self.numChannels, self.numCrops), dtype=np.uint8)#np.float32)
        # img1 = Image.fromarray(image.astype(np.uint8), 'RGB')
        # img.save('my.png')
        # img1.show()
        for row in range(3):
            for col in range(3):
                x_start = crop_x + col * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                # Put the crop in the list of pieces randomly according to the number picked
                final_crops[:, :, :, self.maxHammingSet[perm_index, row * 3 + col]] = \
                    image[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
                # img1.close()
                # img1 = Image.fromarray(final_crops.astype(np.uint8)[:,:,:,self.maxHammingSet[perm_index, row * 3 + col]], 'RGB')
                # img.save('my.png')
                # img1.show()
        return final_crops, perm_index

    def color_channel_jitter(self, image):
        """
        Explain
        """
        # Determine the dimensions of the array, minus the crop around the border
        # of 4 pixels (threshold margin due to 2 pixel jitter)
        x_dim = image.shape[0] - self.colorJitter * 2
        y_dim = image.shape[1] - self.colorJitter * 2
        # Determine the jitters in all directions
        R_xjit = random.randrange(self.colorJitter * 2 + 1)
        R_yjit = random.randrange(self.colorJitter * 2 + 1)
        # Seperate the colour channels
        return_array = np.empty((x_dim, y_dim, 3), np.float32)
        for colour_channel in range(3):
            return_array[:, :, colour_channel] = \
                image[R_xjit:x_dim +R_xjit, R_yjit:y_dim + R_yjit, colour_channel]
        return return_array
