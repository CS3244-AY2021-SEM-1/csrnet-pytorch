import numpy as np
import cv2
import os
import random
import pandas as pd
import h5py


class ImageDataLoader():
    def __init__(self, data_path, shuffle=False, pre_load=False):
        """
        This class returns an iterable object
        Each element in the iterable object is a dictionary containing:
            image: the image
            gt_density: the ground truth density

        Args:
            - data_path: consolidated files path
            - pre_load: if true, all training and validation images are loaded into CPU RAM 
            for faster processing. This avoids frequent file reads. Use this only for small datasets.
        """

        self.data_path = data_path
        self.pre_load = pre_load
        self.shuffle = shuffle
        if shuffle: random.seed(2468)

        self.data_files = [os.path.join(data_path, filename) for filename in os.listdir(data_path)
                           if os.path.isfile(os.path.join(data_path, filename))]
        #self.data_files.sort()

        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = list(range(0, self.num_samples))
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                blob = {}
                f = h5py.File(fname, "r")

                img = f['image'][()]
                blob['data'] = img.reshape((1, 3, img.shape[0], img.shape[1]))  # might want to change this to a tensor object in future

                den = f['density'][()]
                blob['gt_density'] = den.reshape((1, 1, den.shape[0], den.shape[1]))  # might want to change this to a tensor object in future

                blob['fname'] = fname
                self.blob_list[idx] = blob

                idx += 1
                if idx % 100 == 0:
                    print(f'Loaded {idx} / {self.num_samples} files')

            print(f'Completed Loading {idx} files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                fname = files[idx]
                blob = {}
                f = h5py.File(fname, "r")

                img = f['image'][()]
                den = f['density'][()]

                blob['data'] = img.reshape(1, 3, img.shape[0], img.shape[1])        # might want to change this to a tensor object in future
                blob['gt_density'] = den.reshape(1, 1, img.shape[0], img.shape[1])  # might want to change this to a tensor object in future
                #blob['fname'] = fname
                self.blob_list[idx] = blob

            yield blob

    def get_num_samples(self):
        return self.num_samples