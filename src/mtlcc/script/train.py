from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import shutil
import argparse

import csv
import numpy as np

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

import os
import configparser

import functools
import time

from azureml.core.run import Run

# Get run when running in remote
if 'run' not in locals():
    run = Run.get_context()

#
# define constants
#
# file/folders patterns
MODEL_GRAPH_NAME = "graph.meta"
TRAINING_IDS_IDENTIFIER = "train"
TESTING_IDS_IDENTIFIER = "test"
EVAL_IDS_IDENTIFIER = "eval"

MODEL_CFG_FILENAME = "params.ini"
MODEL_CFG_FLAGS_SECTION = "flags"
MODEL_CFG_MODEL_SECTION = "model"
MODEL_CFG_MODEL_KEY = "model"

MODEL_CHECKPOINT_NAME = "model.ckpt"
TRAINING_SUMMARY_FOLDER_NAME = "train"
TESTING_SUMMARY_FOLDER_NAME = "test"
ADVANCED_SUMMARY_COLLECTION_NAME="advanced_summaries"

MASK_FOLDERNAME="mask"
GROUND_TRUTH_FOLDERNAME="ground_truth"
PREDICTION_FOLDERNAME="prediction"
LOSS_FOLDERNAME="loss"
CONFIDENCES_FOLDERNAME="confidences"
TRUE_PRED_FILENAME="truepred.npy"

graph_created_flag = False

#
# define externals functions
#
class ConvLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):
      """A LSTM cell with convolutions instead of multiplications.
      Reference:
        Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
      """

      def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
        super(ConvLSTMCell, self).__init__(_reuse=reuse)
        self._kernel = kernel
        self._filters = filters
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalize = normalize
        self._peephole = peephole
        if data_format == 'channels_last':
            self._size = tf.TensorShape((shape[0],shape[1],self._filters))
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape((self._filters, shape[0],shape[1]))
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')

      @property
      def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

      @property
      def output_size(self):
        return self._size

      def call(self, x, state):
        c, h = state

        x = tf.concat([x, h], axis=self._feature_axis)
        n = x.shape[-1].value
        m = 4 * self._filters if self._filters > 1 else 4
        W = tf.compat.v1.get_variable('kernel', tf.TensorShape((self._kernel[0],self._kernel[1],n,m)))
        y = convolution(x, W, data_format=self._data_format)
        if not self._normalize:
          y += tf.compat.v1.get_variable('bias', [m], initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

        if self._peephole:
          i += tf.compat.v1.get_variable('W_ci', c.shape[1:]) * c
          f += tf.compat.v1.get_variable('W_cf', c.shape[1:]) * c

        if self._normalize:
          j = tf.contrib.layers.layer_norm(j)
          i = tf.contrib.layers.layer_norm(i)
          f = tf.contrib.layers.layer_norm(f)

        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
          o += tf.compat.v1.get_variable('W_co', c.shape[1:]) * c

        if self._normalize:
          o = tf.contrib.layers.layer_norm(o)
          c = tf.contrib.layers.layer_norm(c)

        o = tf.sigmoid(o)
        h = o * self._activation(c)

        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state


class ConvGRUCell(tf.compat.v1.nn.rnn_cell.RNNCell):
      """A GRU cell with convolutions instead of multiplications."""

      def __init__(self, shape, filters, kernel, activation=tf.tanh, normalize=True, data_format='channels_last', reuse=None):
        super(ConvGRUCell, self).__init__(_reuse=reuse)
        self._filters = filters
        self._kernel = kernel
        self._activation = activation
        self._normalize = normalize
        if data_format == 'channels_last':
            self._size = tf.TensorShape((shape[0], shape[1], self._filters))
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape((self._filters, shape[0], shape[1]))
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')

      @property
      def state_size(self):
        return self._size

      @property
      def output_size(self):
        return self._size

      def call(self, x, h):
        channels = x.shape[self._feature_axis].value

        with tf.compat.v1.variable_scope('gates'):
          inputs = tf.concat([x, h], axis=self._feature_axis)
          n = channels + self._filters
          m = 2 * self._filters if self._filters > 1 else 2
          W = tf.compat.v1.get_variable('kernel', tf.TensorShape((self._kernel[0],self._kernel[1],n,m)))
          y = convolution(inputs, W, data_format=self._data_format)
          if self._normalize:
            r, u = tf.split(y, 2, axis=self._feature_axis)
            r = tf.contrib.layers.layer_norm(r)
            u = tf.contrib.layers.layer_norm(u)
          else:
            y += tf.compat.v1.get_variable('bias', [m], initializer=tf.ones_initializer())
            r, u = tf.split(y, 2, axis=self._feature_axis)
          r, u = tf.sigmoid(r), tf.sigmoid(u)

        with tf.compat.v1.variable_scope('candidate'):
          inputs = tf.concat([x, r * h], axis=self._feature_axis)
          n = channels + self._filters
          m = self._filters
          W = tf.compat.v1.get_variable('kernel', tf.TensorShape((self._kernel[0],self._kernel[1],n,m)))
          y = convolution(inputs, W, data_format=self._data_format)
          if self._normalize:
            y = tf.contrib.layers.layer_norm(y)
          else:
            y += tf.compat.v1.get_variable('bias', [m], initializer=tf.zeros_initializer())
          h = u * h + (1 - u) * self._activation(y)

        return h, h


class uparser():
      """ defined the Sentinel 2 .tfrecord format """

      def __init__(self):

        self.feature_format = {
          'x250/data': tf.io.FixedLenFeature([], tf.string),
          'x250/shape': tf.io.FixedLenFeature([4], tf.int64),
          'x250aux/data': tf.io.FixedLenFeature([], tf.string),
          'x250aux/shape': tf.io.FixedLenFeature([4], tf.int64),
          'x500/data': tf.io.FixedLenFeature([], tf.string),
          'x500/shape': tf.io.FixedLenFeature([4], tf.int64),
          'dates/doy': tf.io.FixedLenFeature([], tf.string),
          'dates/year': tf.io.FixedLenFeature([], tf.string),
          'dates/shape': tf.io.FixedLenFeature([1], tf.int64),
          'labels/data': tf.io.FixedLenFeature([], tf.string),
          'labels/shape': tf.io.FixedLenFeature([4], tf.int64)
        }

        return None

      def write(self, filename, x250ds, x500ds, doy, year, labelsds):
        # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

        #         writer = tf.python_io.TFRecordWriter(filename)
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(filename, options=options)

        x250 = x250ds.astype(np.int64)
        x500 = x500ds.astype(np.int64)
        doy = doy.astype(np.int64)
        year = year.astype(np.int64)
        labels = labelsds.astype(np.int64)

        # Create a write feature
        feature = {
          'x250/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x250.tobytes()])),
          'x250/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x250.shape)),
          'x500/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x500.tobytes()])),
          'x500/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x500.shape)),
          'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
          'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
          'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
          'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
          'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

      def parse_example_bands(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """
        feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

        # decode and reshape
        x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

        x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                            tf.cast(feature[0]['labels/shape'], tf.int32))

        return x250, x500, doy, year, labels

      def parse_example_tempCNN(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """
        feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

        # decode and reshape
        x = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                            tf.cast(feature[0]['labels/shape'], tf.int32))

        return x, labels

      def parse_example_bandsaux(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """

        feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

        # decode and reshape
        x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))
        x250aux = tf.reshape(tf.decode_raw(feature[0]['x250aux/data'], tf.int64),
                             tf.cast(feature[0]['x250aux/shape'], tf.int32))

        x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                            tf.cast(feature[0]['labels/shape'], tf.int32))

        return x250, x250aux, x500, doy, year, labels

      def parse_example_bandswoblue(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """
        feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

        # decode and reshape
        x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

        x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                            tf.cast(feature[0]['labels/shape'], tf.int32))

        x500 = x500[:, :, :, 1:5]

        return x250, x500, doy, year, labels

      def parse_example_bands250m(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """
        feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

        # decode and reshape
        x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                            tf.cast(feature[0]['labels/shape'], tf.int32))

        x500 = x250[:, :, :, 0:1]

        return x250, x500, doy, year, labels

      def read(self, filenames):
        """ depricated! """

        if isinstance(filenames, list):
          filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        elif isinstance(filenames, tf.FIFOQueue):
          filename_queue = filenames
        else:
          print ("please insert either list or tf.FIFOQueue")

        reader = tf.TFRecordReader()
        f, serialized_example = reader.read(filename_queue)

        feature = tf.parse_single_example(serialized_example, features=self.feature_format)

        # decode and reshape
        x250 = tf.reshape(tf.decode_raw(feature['x250/data'], tf.int64), tf.cast(feature['x250/shape'], tf.int32))
        x500 = tf.reshape(tf.decode_raw(feature['x500/data'], tf.int64), tf.cast(feature['x500/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

        return x250, x500, doy, year, labels

      def read_and_return(self, filename):
        """ depricated! """

        # get feature operation containing
        feature_op = self.read([filename])

        with tf.Session() as sess:
          tf.global_variables_initializer()

          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)

          return sess.run(feature_op)

      def get_shapes(self, sample):
        print("reading shape of data using the sample " + sample)
        data = self.read_and_return(sample)
        return [tensor.shape for tensor in data]

      def tfrecord_to_pickle(self, tfrecordname, picklename):
        import cPickle as pickle

        reader = tf.TFRecordReader()

        # read serialized representation of *.tfrecord
        filename_queue = tf.train.string_input_producer([tfrecordname], num_epochs=None)
        filename_op, serialized_example = reader.read(filename_queue)
        feature = self.parse_example(serialized_example)

        with tf.Session() as sess:
          sess.run([tf.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)

          feature = sess.run(feature)

          coord.request_stop()
          coord.join(threads)

        pickle.dump(feature, open(picklename, "wb"), protocol=2)


#
# define dataset functions
#
class Dataset():
    """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

    def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False,
                 experiment="bands",
                 reference="MCD12Q1v6stable01to15_LCProp2_major", step="training"):
        self.verbose = verbose

        self.augment = augment

        self.experiment = experiment
        self.reference = reference

        # parser reads serialized tfrecords file and creates a feature object
        parser = uparser()
        if self.experiment == "bands" or self.experiment == "bandswodoy":
            self.parsing_function = parser.parse_example_bands
        elif self.experiment == "indices":
            self.parsing_function = parser.parse_example_bandsaux
        elif self.experiment == "bandsaux":
            self.parsing_function = parser.parse_example_bandsaux
        elif self.experiment == "all":
            self.parsing_function = parser.parse_example_bandsaux
        elif self.experiment == "bandswoblue":
            self.parsing_function = parser.parse_example_bandswoblue
        elif self.experiment == "bands250m" or self.experiment == "evi2":
            self.parsing_function = parser.parse_example_bands250m

        self.temp_samples = temporal_samples
        self.section = section
        self.step = step

        dataroot = datadir

        # csv list of geotransforms of each tile: tileid, xmin, xres, 0, ymax, 0, -yres, srid
        # use querygeotransform.py or querygeotransforms.sh to generate csv
        # fills dictionary:
        # geotransforms[<tileid>] = (xmin, xres, 0, ymax, 0, -yres)
        # srid[<tileid>] = srid
        self.geotransforms = dict()
        # https://en.wikipedia.org/wiki/Spatial_reference_system#Identifier
        self.srids = dict()
        with file_io.FileIO(os.path.join(dataroot, "geotransforms.csv"), 'r') as f:  # gcp
            # with open(os.path.join(dataroot, "geotransforms.csv"),'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # float(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]))
                self.geotransforms[ str(row[ 0 ]) ] = (
                    float(row[ 1 ]), float(row[ 2 ]), int(row[ 3 ]), float(row[ 4 ]), int(row[ 5 ]), float(row[ 6 ]))
                self.srids[ str(row[ 0 ]) ] = int(row[ 7 ])

        classes = os.path.join(dataroot, "classes_" + reference + ".txt")
        with file_io.FileIO(classes, 'r') as f:  # gcp
            # with open(classes, 'r') as f:
            classes = f.readlines()

        self.ids = list()
        self.classes = list()
        for row in classes:
            row = row.replace("\n", "")
            if '|' in row:
                id, cl = row.split('|')
                self.ids.append(int(id))
                self.classes.append(cl)

        cfgpath = os.path.join(dataroot, "dataset.ini")
        print(cfgpath)
        # load dataset configs
        datacfg = configparser.ConfigParser()
        with file_io.FileIO(cfgpath, 'r') as f:  # gcp
            datacfg.read_file(f)

        cfg = datacfg[ section ]

        self.tileidfolder = os.path.join(dataroot, "tileids")
        self.datadir = os.path.join(dataroot, cfg[ "datadir" ])

        assert 'pix250' in cfg.keys()
        assert 'nobs' in cfg.keys()
        assert 'nbands250' in cfg.keys()
        assert 'nbands500' in cfg.keys()

        self.tiletable = cfg[ "tiletable" ]

        self.nobs = int(cfg[ "nobs" ])

        self.expected_shapes = self.calc_expected_shapes(int(cfg[ "pix250" ]),
                                                         int(cfg[ "nobs" ]),
                                                         int(cfg[ "nbands250" ]),
                                                         int(cfg[ "nbands500" ]),
                                                         )

        # expected datatypes as read from disk
        self.expected_datatypes = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)

    def calc_expected_shapes(self, pix250, nobs, bands250, bands500):
        pix250 = pix250
        pix500 = pix250 / 2
        x250shape = (nobs, pix250, pix250, bands250)
        x500shape = (nobs, pix500, pix500, bands500)
        doyshape = (nobs,)
        yearshape = (nobs,)
        labelshape = (nobs, pix250, pix250)

        return [ x250shape, x500shape, doyshape, yearshape, labelshape ]

    def transform_labels_training(self, feature):
        """
        1. prepare for the estimator
        """

        x250, x500, doy, year, labels = feature

        return (x250, x500, doy, year), labels

    def transform_labels_evaluation(self, feature):
        """
        1. take only first labelmap, as labels are not supposed to change
        """

        x250, x500, doy, year, labels = feature

        # take first label time [46,24,24] -> [24,24]
        # labels are not supposed to change over the time series
        labels = labels[ 0 ]

        return x250, x500, doy, year, labels

    def normalize_old(self, feature):

        x250, x500, doy, year, labels = feature
        x250 = tf.scalar_mul(1e-4, tf.cast(x250, tf.float32))
        x500 = tf.scalar_mul(1e-4, tf.cast(x500, tf.float32))

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year / cancel year

        return x250, x500, doy, year, labels

    def normalize_bands250m(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        # normal minx/max domain
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # 250m
        # SR
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)
        norm250m = [ x_normed_red, x_normed_NIR ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        # cancel the effect of 500m
        x500 = tf.cast(x500, tf.float32) - tf.cast(x500, tf.float32)

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, x500, doy, year, labels

    def normalize_evi2(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # indices
        fixed_range = [ [ -10000, 10000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_evi2 = normalize_fixed(x250[ :, :, :, 2 ], fixed_range, normed_range)

        norm250m = [ x_normed_evi2 ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        # cancel effect 500m
        x500 = tf.cast(x500, tf.float32) - tf.cast(x500, tf.float32)

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)

        return norm250m, x500, doy, year, labels

    def normalize_bands(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        # normal minx/max domain
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # 250m
        # SR
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)
        norm250m = [ x_normed_red, x_normed_NIR ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        # 500m
        x_normed_blue = normalize_fixed(x500[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[ :, :, :, 1 ], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[ :, :, :, 4 ], fixed_range, normed_range)

        norm500m = [ x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3 ]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [ 1, 2, 3, 0 ])

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_bandswoblue(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        # normal minx/max domain
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # 250m
        # SR
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)
        norm250m = [ x_normed_red, x_normed_NIR ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        # 500m
        x_normed_green = normalize_fixed(x500[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[ :, :, :, 1 ], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[ :, :, :, 3 ], fixed_range, normed_range)

        norm500m = [ x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3 ]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [ 1, 2, 3, 0 ])

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_bandsaux(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x250aux, x500, doy, year, labels = feature

        x250aux = tf.tile(x250aux, [ self.nobs, 1, 1, 1 ])

        # normal minx/max domain
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # SR
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        # 250m
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)

        # 500m
        x_normed_blue = normalize_fixed(x500[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[ :, :, :, 1 ], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[ :, :, :, 4 ], fixed_range, normed_range)

        # bio 01
        fixed_range = [ [ -290, 320 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio01 = normalize_fixed(x250aux[ :, :, :, 0 ], fixed_range, normed_range)

        # bio 12
        fixed_range = [ [ 0, 11401 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio12 = normalize_fixed(x250aux[ :, :, :, 1 ], fixed_range, normed_range)

        # elevation
        fixed_range = [ [ -444, 8806 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_ele = normalize_fixed(x250aux[ :, :, :, 2 ], fixed_range, normed_range)

        norm250m = [ x_normed_red, x_normed_NIR, x_normed_bio01, x_normed_bio12, x_normed_ele ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        norm500m = [ x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3 ]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [ 1, 2, 3, 0 ])

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_indices(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x250aux, x500, doy, year, labels = feature

        # normed values
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # SR
        # 250m
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)

        # 500m
        x_normed_blue = normalize_fixed(x500[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[ :, :, :, 1 ], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[ :, :, :, 4 ], fixed_range, normed_range)

        # indices
        fixed_range = [ [ -10000, 10000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_evi2 = normalize_fixed(x250[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_ndwi = normalize_fixed(x250[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_ndii1 = normalize_fixed(x250[ :, :, :, 4 ], fixed_range, normed_range)
        x_normed_ndii2 = normalize_fixed(x250[ :, :, :, 5 ], fixed_range, normed_range)
        x_normed_ndsi = normalize_fixed(x250[ :, :, :, 6 ], fixed_range, normed_range)

        norm250m = [ x_normed_red, x_normed_NIR, x_normed_evi2, x_normed_ndwi, x_normed_ndii1, x_normed_ndii2,
                     x_normed_ndsi ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        norm500m = [ x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3 ]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [ 1, 2, 3, 0 ])

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_all(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x250aux, x500, doy, year, labels = feature

        x250aux = tf.tile(x250aux, [ self.nobs, 1, 1, 1 ])

        # normed values
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # SR
        # 250m
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)

        # 500m
        x_normed_blue = normalize_fixed(x500[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[ :, :, :, 1 ], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[ :, :, :, 4 ], fixed_range, normed_range)

        # bio 01
        fixed_range = [ [ -290, 320 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio01 = normalize_fixed(x250aux[ :, :, :, 0 ], fixed_range, normed_range)

        # bio 12
        fixed_range = [ [ 0, 11401 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio12 = normalize_fixed(x250aux[ :, :, :, 1 ], fixed_range, normed_range)

        # elevation
        fixed_range = [ [ -444, 8806 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_ele = normalize_fixed(x250aux[ :, :, :, 2 ], fixed_range, normed_range)

        # indices
        fixed_range = [ [ -10000, 10000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_evi2 = normalize_fixed(x250[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_ndwi = normalize_fixed(x250[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_ndii1 = normalize_fixed(x250[ :, :, :, 4 ], fixed_range, normed_range)
        x_normed_ndii2 = normalize_fixed(x250[ :, :, :, 5 ], fixed_range, normed_range)
        x_normed_ndsi = normalize_fixed(x250[ :, :, :, 6 ], fixed_range, normed_range)

        norm250m = [ x_normed_red, x_normed_NIR, x_normed_bio01, x_normed_bio12, x_normed_ele, x_normed_evi2,
                     x_normed_ndwi,
                     x_normed_ndii1, x_normed_ndii2, x_normed_ndsi ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        norm500m = [ x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3 ]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [ 1, 2, 3, 0 ])

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_bandswodoy(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[ :, 0 ], 1), tf.expand_dims(current_range[ :, 1 ],
                                                                                                1)
            normed_min, normed_max = tf.expand_dims(normed_range[ :, 0 ], 1), tf.expand_dims(normed_range[ :, 1 ], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        # normal minx/max domain
        fixed_range = [ [ -100, 16000 ] ]
        fixed_range = np.array(fixed_range)
        normed_range = [ [ 0, 1 ] ]
        normed_range = np.array(normed_range)

        # 250m
        # SR
        x_normed_red = normalize_fixed(x250[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[ :, :, :, 1 ], fixed_range, normed_range)
        norm250m = [ x_normed_red, x_normed_NIR ]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [ 1, 2, 3, 0 ])

        # 500m
        x_normed_blue = normalize_fixed(x500[ :, :, :, 0 ], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[ :, :, :, 1 ], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[ :, :, :, 2 ], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[ :, :, :, 3 ], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[ :, :, :, 4 ], fixed_range, normed_range)

        norm500m = [ x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3 ]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [ 1, 2, 3, 0 ])

        doy = tf.cast(doy, tf.float32) - tf.cast(doy, tf.float32)

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def augment(self, feature):

        x250, x500, doy, year, labels = feature

        ## Flip UD
        # roll the dice
        condition = tf.less(tf.random_uniform(shape=[ ], minval=0., maxval=1., dtype=tf.float32), 0.5)

        # flip
        x250 = tf.cond(condition, lambda: tf.reverse(x250, axis=[ 1 ]), lambda: x250)
        x500 = tf.cond(condition, lambda: tf.reverse(x500, axis=[ 1 ]), lambda: x500)
        labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[ 1 ]), lambda: labels)

        ## Flip LR
        # roll the dice
        condition = tf.less(tf.random_uniform(shape=[ ], minval=0., maxval=1., dtype=tf.float32), 0.5)

        # flip
        x250 = tf.cond(condition, lambda: tf.reverse(x250, axis=[ 2 ]), lambda: x250)
        x500 = tf.cond(condition, lambda: tf.reverse(x500, axis=[ 2 ]), lambda: x500)
        labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[ 2 ]), lambda: labels)

        return x250, x500, doy, year, labels

    def temporal_sample(self, feature):
        """ randomy choose <self.temp_samples> elements from temporal sequence """

        n = self.temp_samples

        # skip if not specified
        if n is None:
            return feature

        x250, x500, doy, year, labels = feature

        max_obs = self.nobs

        shuffled_range = tf.random.shuffle(tf.range(max_obs))[ 0:n ]

        idxs = -tf.nn.top_k(-shuffled_range, k=n).values

        x250 = tf.gather(x250, idxs)
        x500 = tf.gather(x500, idxs)
        doy = tf.gather(doy, idxs)
        year = tf.gather(year, idxs)

        return x250, x500, doy, year, labels

    def addIndices(self, features):

        def NDVI(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def EVI2(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
            nd = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000))
            nd_inf = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000 + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDWI(a, b):  # 10000*(double)(nir[ii]-swir1[ii]) / (double)(nir[ii]+swir1[ii]);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDSI(a, b):  # 10000*(double)(green[ii]-swir2[ii]) / (double)(green[ii]+swir2[ii]);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDII1(a, b):  # 10000*(double)(nir[ii]-swir2[ii]) / (double)(nir[ii]+swir2[ii])
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDII2(a, b):  # 10000*(double)(nir[ii]-swir3[ii]) / (double)(nir[ii]+swir3[ii]);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def resize(tensor, new_height, new_width):
            t = tf.shape(tensor)[ 0 ]
            h = tf.shape(tensor)[ 1 ]
            w = tf.shape(tensor)[ 2 ]
            d = tf.shape(tensor)[ 3 ]

            # stack batch on times to fit 4D requirement of resize_tensor
            stacked_tensor = tf.reshape(tensor, [ t, h, w, d ])
            reshaped_stacked_tensor = tf.compat.v1.image.resize_images(stacked_tensor, size=(new_height, new_width))
            return tf.reshape(reshaped_stacked_tensor, [ t, new_height, new_width, d ])

        x250, x250aux, x500, doy, year, labels = features

        px = tf.shape(x250)[ 2 ]

        x250 = tf.cast(x250, tf.float32)
        x500 = tf.cast(x500, tf.float32)

        x500_r = tf.identity(resize(x500, px, px))

        ndvi = NDVI(x250[ :, :, :, 1 ], x250[ :, :, :, 0 ])
        evi2 = EVI2(x250[ :, :, :, 1 ], x250[ :, :, :, 0 ])
        ndwi = NDWI(x250[ :, :, :, 1 ], x500_r[ :, :, :, 2 ])
        ndii1 = NDII1(x250[ :, :, :, 1 ], x500_r[ :, :, :, 3 ])
        ndii2 = NDII2(x250[ :, :, :, 1 ], x500_r[ :, :, :, 4 ])
        ndsi = NDSI(x500_r[ :, :, :, 1 ], x500_r[ :, :, :, 3 ])

        indices250m = [ evi2, ndwi, ndii1, ndii2, ndsi ]

        x250indices = tf.stack(indices250m)
        x250indices = tf.transpose(x250indices, [ 1, 2, 3, 0 ])

        x250m = tf.concat([ x250, x250indices ], axis=3)

        return x250m, x250aux, x500, doy, year, labels

    def addIndices250m(self, features):

        def EVI2(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
            nd = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000))
            nd_inf = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000 + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        x250, x500, doy, year, labels = features

        x250 = tf.cast(x250, tf.float32)

        evi2 = EVI2(x250[ :, :, :, 1 ], x250[ :, :, :, 0 ])

        # indices250m = [evi2]
        indices250m = [ evi2 ]

        x250indices = tf.stack(indices250m)
        x250indices = tf.transpose(x250indices, [ 1, 2, 3, 0 ])

        x250m = tf.concat([ x250, x250indices ], axis=3)

        return x250m, x500, doy, year, labels

    def MCD12Q1v6raw_LCType1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 0 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable_LCType1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 1 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6raw_LCProp1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 2 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable_LCProp1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 2 ]

        return x250, x500, doy, year, labels

    # def MCD12Q1v6raw_LCProp2(self, feature):
    #
    #   x250, x500, doy, year, labels = feature
    #
    #   labels = labels[ :, :, :, 4 ]
    #
    #   return x250, x500, doy, year, labels
    #
    # def MCD12Q1v6stable01to15_LCProp2(self, feature):
    #
    #   x250, x500, doy, year, labels = feature
    #
    #   labels = labels[ :, :, :, 5 ]
    #
    #   return x250, x500, doy, year, labels
    #
    # def MCD12Q1v6stable01to03_LCProp2(self, feature):
    #
    #   x250, x500, doy, year, labels = feature
    #
    #   labels = labels[ :, :, :, 6]
    #
    #   return x250, x500, doy, year, labels

    def ESAraw(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 7 ]

        return x250, x500, doy, year, labels

    def ESAstable(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 8 ]

        return x250, x500, doy, year, labels

    def Copernicusraw(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 9 ]

        return x250, x500, doy, year, labels

    def Copernicusraw_fraction(self, feature):

        x250, x500, doy, year, labels = feature

        labels = tf.argmax(labels, axis=3)

        return x250, x500, doy, year, labels

    def watermask(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 10 ]

        return x250, x500, doy, year, labels

    def Copernicusnew_cf2others(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 11 ]

        return x250, x500, doy, year, labels

    def merge_datasets2own(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 12 ]

        return x250, x500, doy, year, labels

    def merge_datasets2HuHu(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 13 ]

        return x250, x500, doy, year, labels

    def merge_datasets2Tsendbazaretal2maps(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 14 ]

        return x250, x500, doy, year, labels

    def merge_datasets2Tsendbazaretal3maps(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 15 ]

        return x250, x500, doy, year, labels

    def merge_datasets2Tsendbazaretal3maps(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 15 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6raw_LCProp2_major(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 16 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable01to15_LCProp2_major(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 17 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable01to03_LCProp2_major(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 18 ]

        return x250, x500, doy, year, labels

    def mapbiomas_fraction(self, feature):

        x250, x500, doy, year, labels = feature

        labels = tf.argmax(labels, axis=3)

        return x250, x500, doy, year, labels

    def tilestatic(self, feature):

        x250, x500, doy, year, labels = feature

        labels = tf.tile(labels, [ self.nobs, 1, 1, 1 ])

        return x250, x500, doy, year, labels

    def get_ids(self, partition, fold=0):

        def readids(path):
            with file_io.FileIO(path, 'r') as f:  # gcp
                # with open(path, 'r') as f:
                lines = f.readlines()
            ##ac            return [int(l.replace("\n", "")) for l in lines]
            return [ str(l.replace("\n", "")) for l in lines ]

        traintest = "{partition}_fold{fold}.tileids"
        eval = "{partition}.tileids"

        if partition == 'train':
            # e.g. train240_fold0.tileids
            path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
            return readids(path)
        elif partition == 'test':
            # e.g. test240_fold0.tileids
            path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
            return readids(path)
        elif partition == 'eval':
            # e.g. eval240.tileids
            path = os.path.join(self.tileidfolder, eval.format(partition=partition))
            return readids(path)
        else:
            raise ValueError("please provide valid partition (train|test|eval)")

    def create_tf_dataset(self, partition, fold, batchsize, prefetch_batches=None, num_batches=-1, threads=8,
                          drop_remainder=False, overwrite_ids=None):

        # set of ids as present in database of given partition (train/test/eval) and fold (0-9)
        allids = self.get_ids(partition=partition, fold=fold)

        # set of ids present in local folder (e.g. 1.tfrecord)
        # tiles = os.listdir(self.datadir)
        tiles = file_io.get_matching_files(os.path.join(self.datadir, '*.gz'))
        tiles = [ os.path.basename(t) for t in tiles ]

        if tiles[ 0 ].endswith(".gz"):
            compression = "GZIP"
            ext = ".gz"
        else:
            compression = ""
            ext = ".tfrecord"

        downloaded_ids = [ str(t.replace(".gz", "").replace(".tfrecord", "")) for t in tiles ]

        allids = [ i.strip() for i in allids ]

        # intersection of available ids and partition ods
        if overwrite_ids is None:
            ids = list(set(downloaded_ids).intersection(allids))
        else:
            print("overwriting data ids! due to manual input")
            ids = overwrite_ids

        filenames = [ os.path.join(self.datadir, str(id) + ext) for id in ids ]

        if self.verbose:
            print(
                "dataset: {}, partition: {}, fold:{} {}/{} tiles downloaded ({:.2f} %)".format(self.section, partition,
                                                                                               fold,
                                                                                               len(ids), len(allids),
                                                                                               len(ids) / float(
                                                                                                   len(allids)) * 100))

        def mapping_function(serialized_feature):
            # read data from .tfrecords
            feature = self.parsing_function(serialized_example=serialized_feature)
            # sample n times out of the timeseries
            feature = self.temporal_sample(feature)
            # indices
            if self.experiment == "indices" or self.experiment == "all": feature = self.addIndices(feature)
            if self.experiment == "evi2": feature = self.addIndices250m(feature)

            # perform data normalization [0,1000] -> [0,1]
            if self.experiment == "bands250m": feature = self.normalize_bands250m(feature)
            if self.experiment == "bands": feature = self.normalize_bands(feature)
            if self.experiment == "bandswoblue": feature = self.normalize_bandswoblue(feature)
            if self.experiment == "bandsaux": feature = self.normalize_bandsaux(feature)
            if self.experiment == "indices": feature = self.normalize_indices(feature)
            if self.experiment == "all": feature = self.normalize_all(feature)
            if self.experiment == "evi2": feature = self.normalize_evi2(feature)
            if self.experiment == "bandswodoy": feature = self.normalize_bandswodoy(feature)

            feature = self.tilestatic(feature)

            if self.reference == "MCD12Q1v6raw_LCType1": feature = self.MCD12Q1v6raw_LCType1(feature)
            if self.reference == "MCD12Q1v6raw_LCProp1": feature = self.MCD12Q1v6raw_LCProp1(feature)
            if self.reference == "MCD12Q1v6raw_LCProp2": feature = self.MCD12Q1v6raw_LCProp2(feature)
            if self.reference == "MCD12Q1v6raw_LCProp2_major": feature = self.MCD12Q1v6raw_LCProp2_major(feature)
            if self.reference == "MCD12Q1v6stable_LCType1": feature = self.MCD12Q1v6stable_LCType1(feature)
            if self.reference == "MCD12Q1v6stable_LCProp1": feature = self.MCD12Q1v6stable_LCProp1(feature)
            if self.reference == "MCD12Q1v6stable01to15_LCProp2": feature = self.MCD12Q1v6stable01to15_LCProp2(feature)
            if self.reference == "MCD12Q1v6stable01to03_LCProp2": feature = self.MCD12Q1v6stable01to03_LCProp2(feature)
            if self.reference == "MCD12Q1v6stable01to15_LCProp2_major": feature = self.MCD12Q1v6stable01to15_LCProp2_major(
                feature)
            if self.reference == "MCD12Q1v6stable01to03_LCProp2_major": feature = self.MCD12Q1v6stable01to03_LCProp2_major(
                feature)
            if self.reference == "ESAraw": feature = self.ESAraw(feature)
            if self.reference == "ESAstable": feature = self.ESAstable(feature)
            if self.reference == "Copernicusraw": feature = self.Copernicusraw(feature)
            if self.reference == "Copernicusraw_fraction": feature = self.Copernicusraw_fraction(feature)
            if self.reference == "Copernicusnew_cf2others": feature = self.Copernicusnew_cf2others(feature)
            if self.reference == "merge_datasets2own": feature = self.merge_datasets2own(feature)
            if self.reference == "merge_datasets2HuHu": feature = self.merge_datasets2HuHu(feature)
            if self.reference == "merge_datasets2Tsendbazaretal2maps": feature = self.merge_datasets2Tsendbazaretal2maps(
                feature)
            if self.reference == "merge_datasets2Tsendbazaretal3maps": feature = self.merge_datasets2Tsendbazaretal3maps(
                feature)
            if self.reference == "mapbiomas_fraction": feature = self.mapbiomas_fraction(feature)
            if self.reference == "watermask": feature = self.watermask(feature)

            # perform data augmentation
            if self.augment: feature = self.augment(feature)

            if self.step == "training": feature = self.transform_labels_training(feature)
            if not self.step == "training": feature = self.transform_labels_evaluation(feature)

            return feature

        if num_batches > 0:
            filenames = filenames[ 0:num_batches * batchsize ]

        # shuffle sequence of filenames
        if partition == 'train':
            filenames = tf.random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, compression_type=compression, num_parallel_reads=threads)

        dataset = dataset.map(mapping_function, num_parallel_calls=threads)

        # repeat forever until externally stopped
        dataset = dataset.repeat()

        if drop_remainder:
            dataset = dataset.apply(tf.data.batch_and_drop_remainder(int(batchsize)))
        else:
            dataset = dataset.batch(int(batchsize))

        if prefetch_batches is not None:
            dataset = dataset.prefetch(prefetch_batches)

        # model shapes are expected shapes of the data stacked as batch
        output_shape = [ ]
        for shape in self.expected_shapes:
            output_shape.append(tf.TensorShape((batchsize,) + shape))

        return dataset, output_shape, self.expected_datatypes, filenames


#
# define input functions
#
def input_fn_train_singleyear(args, mode):
  """Reads TFRecords and returns the features and labels."""

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=args.train_on[0],
                           experiment=args.experiment, reference=args.reference)

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  # iterator = tfdataset.dataset.make_one_shot_iterator()
  iterator = tf.compat.v1.data.make_initializable_iterator(tfdataset)
  #
  tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

  features, labels = iterator.get_next()

  return features, labels


def input_fn_train_multiyear(args, mode):
  """Reads TFRecords and returns the features and labels."""
  datasets_dict = dict()

  for section in args.train_on.split(' '):
    datasets_dict[section] = dict()

    dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=section,
                      experiment=args.experiment, reference=args.reference)

    if mode == tf.estimator.ModeKeys.TRAIN:
      partition = TRAINING_IDS_IDENTIFIER
    elif mode == tf.estimator.ModeKeys.EVAL:
      partition = TESTING_IDS_IDENTIFIER
    elif mode == tf.estimator.ModeKeys.PREDICT:
      partition = EVAL_IDS_IDENTIFIER

    datasets_dict[ section ][ partition ] = dict()

    # datasets_dict[section][partition] = dict()
    tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                           args.fold,
                                                           args.batchsize,
                                                           prefetch_batches=args.prefetch,
                                                           num_batches=args.limit_batches)

    # iterator = tfdataset.dataset.make_one_shot_iterator()
    datasets_dict[section][partition]["tfdataset"] = tfdataset

  if len(args.train_on.split(' ')) > 1:

    ds = datasets_dict[args.train_on.split(' ')[0]][partition]["tfdataset"]

    for section in args.train_on.split(' ')[1:]:
      ds = ds.concatenate(datasets_dict[section][partition]["tfdataset"])

  else:
    ds = datasets_dict[args.train_on.split(' ')[0]][partition]["tfdataset"]

  return ds


def input_fn_eval(args, mode):
  """Reads TFRecords and returns the features and labels."""

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=args.dataset,
                           experiment=args.experiment, reference=args.reference)

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.PREDICT:
    partition = EVAL_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  return tfdataset


def input_filenames(args, mode):
  """Reads TFRecords and returns the features and labels."""

  if args.step == 'training':
    target = args.train_on.split(' ')[0]
  else:
    target = args.dataset

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=target,
                           experiment=args.experiment, reference=args.reference)

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.PREDICT:
    partition = EVAL_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  return filenames, dataset


#
# define model functions
#
## hyper parameters ##
tf.app.flags.DEFINE_string("kernel", "(1,3,3)", "kernel of convolutions")
tf.app.flags.DEFINE_string("classkernel", "(3,3)", "kernelsize of final classification convolution")
tf.app.flags.DEFINE_string("cnn_activation", 'leaky_relu',
                           "activation function for convolutional layers ('relu' or 'leaky_relu')")

tf.app.flags.DEFINE_boolean("bidirectional", True, "Bidirectional Convolutional RNN")
tf.app.flags.DEFINE_integer("convrnn_compression_filters", -1,
                            "number of convrnn compression filters or (default) -1 for no compression")
tf.app.flags.DEFINE_string("convcell", "gru", "Convolutional RNN cell architecture ('gru' (default) or 'lstm')")
tf.app.flags.DEFINE_string("convrnn_kernel", "(3,3)", "kernelsize of recurrent convolution. default (3,3)")
tf.app.flags.DEFINE_integer("convrnn_filters", None, "number of convolutional filters in ConvLSTM/ConvGRU layer")
tf.app.flags.DEFINE_float("recurrent_dropout_i", 1.,
                          "input keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_float("recurrent_dropout_c", 1.,
                          "state keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_integer("convrnn_layers", 1, "number of convolutional recurrent layers")
tf.app.flags.DEFINE_boolean("peephole", False,
                            "use peephole connections at convrnn layer. only for lstm (default False)")
tf.app.flags.DEFINE_boolean("convrnn_normalize", True, "normalize with batchnorm at convrnn layer (default True)")
tf.app.flags.DEFINE_string("aggr_strat", "state",
                           "aggregation strategie to reduce temporal dimension (either default 'state' or 'sum_output' or 'avg_output')")

tf.app.flags.DEFINE_float("learning_rate", None, "Adam learning rate")
tf.app.flags.DEFINE_float("beta1", 0.9, "Adam beta1")
tf.app.flags.DEFINE_float("beta2", 0.999, "Adam beta2")
tf.app.flags.DEFINE_float("epsilon", 1e-8, "Adam epsilon")
# tf.app.flags.DEFINE_float("epsilon", 0.9, "Adam epsilon")

## expected data format ##
tf.app.flags.DEFINE_string("expected_datatypes",
                           "(tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)", "expected datatypes")
tf.app.flags.DEFINE_integer("pix250m", None, "number of 250m pixels")
tf.app.flags.DEFINE_integer("num_classes", 8,
                            "number of classes not counting unknown class -> e.g. 0:uk,1:a,2:b,3:c,4:d -> num_classes 4")

## performance ##
tf.app.flags.DEFINE_boolean("swap_memory", True, "Swap memory between GPU and CPU for recurrent layers")

tf.app.flags.DEFINE_string('experiment', None, 'experiment')
tf.app.flags.DEFINE_string('reference', None, 'reference')
tf.app.flags.DEFINE_string('optimizertype', None, 'experiment')

tf.app.flags.DEFINE_string('datadir', None, 'datadir')
tf.app.flags.DEFINE_string('fold', None, 'fold')
tf.app.flags.DEFINE_string('modeldir', None, 'modeldir')
tf.app.flags.DEFINE_integer('batchsize', None, 'batchsize')
tf.app.flags.DEFINE_string('train_on', None, 'train_on')
tf.app.flags.DEFINE_integer('epochs', None, 'epochs')
tf.app.flags.DEFINE_integer('limit_batches', None, 'limit_batches')
tf.app.flags.DEFINE_string('step', None, 'step')

# #datadir

FLAGS = tf.app.flags.FLAGS

MODEL_CFG_FILENAME = 'params.ini'

ADVANCED_SUMMARY_COLLECTION_NAME="advanced_summaries"


def inference(input, is_train=True, num_classes=None):
    x, sequence_lengths = input

    rnn_output_list = list()
    rnn_state_list = list()

    x_rnn = x
    for j in range(1, FLAGS.convrnn_layers + 1):
        convrnn_kernel = eval(FLAGS.convrnn_kernel)
        x_rnn, state = convrnn_layer(input=x_rnn, is_train=is_train, filter=FLAGS.convrnn_filters,
                                     kernel=convrnn_kernel,
                                     bidirectional=FLAGS.bidirectional, convcell=FLAGS.convcell,
                                     sequence_lengths=sequence_lengths, scope="convrnn" + str(j))
        rnn_output_list.append(x_rnn)
        rnn_state_list.append(state)

    # # concat outputs from cnns and rnns in a dense scheme
    x = tf.concat(rnn_output_list, axis=-1)

    # # take concatenated states of last rnn block (might contain multiple conrnn layers)
    state = tf.concat(rnn_state_list, axis=-1)

    # use the cell state as featuremap for the classification step
    # cell state has dimensions (b,h,w,d) -> classification strategy
    if FLAGS.aggr_strat == 'state':
        class_input = state  # shape (b,h,w,d)
        classkernel = eval(FLAGS.classkernel)
        logits = conv_bn_relu(input=class_input, is_train=is_train, filter=num_classes,
                              kernel=classkernel, dilation_rate=(1, 1), conv_fun=tf.keras.layers.Conv2D,
                              var_scope="class")

    elif (FLAGS.aggr_strat == 'avg_output') or (FLAGS.aggr_strat == 'sum_output'):
        # last rnn output at each time t
        # class_input = x_rnn  # shape (b,t,h,w,d)
        class_input = x  # shape (b,t,h,w,d)

        # kernel = (1,FLAGS.classkernel[0],FLAGS.classkernel[1])
        kernel = (1, eval(FLAGS.classkernel)[0], eval(FLAGS.classkernel)[1])

        # logits for each single timeframe
        logits = conv_bn_relu(input=class_input, is_train=is_train, filter=num_classes, kernel=kernel,
                              dilation_rate=(1, 1, 1), conv_fun=tf.keras.layers.Conv3D, var_scope="class")

        if FLAGS.aggr_strat == 'avg_output':
            # average logit scores at each observation
            # (b,t,h,w,d) -> (b,h,w,d)
            logits = tf.reduce_mean(logits, axis=1)

        elif FLAGS.aggr_strat == 'sum_output':
            # summarize logit scores at each observation
            # the softmax normalization later will normalize logits again
            # (b,t,h,w,d) -> (b,h,w,d)
            logits = tf.reduce_sum(logits, axis=1)

    else:
        raise ValueError("please provide valid aggr_strat flag ('state' or 'avg_output' or 'sum_output')")

    tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, logits)

    return logits


def loss(logits, labels, mask, name):

    loss_per_px = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    _ = tf.identity(loss_per_px, name="loss_per_px")
    _ = tf.identity(mask, name="mask_per_px")

    lpp = tf.boolean_mask(loss_per_px, mask)

    return tf.reduce_mean(lpp, name=name)


def optimize(loss, global_step, otype, name):
    lr = tf.compat.v1.placeholder_with_default(FLAGS.learning_rate, shape=(), name="learning_rate")
    beta1 = tf.compat.v1.placeholder_with_default(FLAGS.beta1, shape=(), name="beta1")
    beta2 = tf.compat.v1.placeholder_with_default(FLAGS.beta2, shape=(), name="beta2")

    if otype == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2,
            epsilon=FLAGS.epsilon
        )
    elif otype == 'nadam':
        optimizer = tf.contrib.opt.NadamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2,
            epsilon=FLAGS.epsilon
        )
    elif otype == "adamW":
        ##TODO - need further work
        optimizer = tf.contrib.opt.AdamWOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2,
            epsilon=FLAGS.epsilon, weight_decay=FLAGS.weight_decay
        )

    # method 1
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        return optimizer.minimize(loss, global_step=global_step, name=name)

    #method 2 - slower source: https://github.com/tensorflow/tensorflow/issues/25057
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # minimize_op = optimizer.minimize(loss=loss, global_step=global_step,name=name)
    # train_op = tf.group(minimize_op, update_ops)

    # return train_op


def eval_oa(logits, labels, mask):

    prediction_scores = tf.nn.softmax(logits=logits, name="prediction_scores")
    predictions = tf.argmax(prediction_scores, 3, name="predictions")

    targets = tf.argmax(labels, 3, name="targets")

    correctly_predicted = tf.equal(tf.boolean_mask(predictions, mask), tf.boolean_mask(targets, mask),
                                   name="correctly_predicted")

    overall_accuracy = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32), name="overall_accuracy")

    ###Single-GPU
    overall_accuracy_sum = tf.Variable(tf.zeros(shape=([]), dtype=tf.float32),
                                       trainable=False,
                                       name="overall_accuracy_result",
                                       collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES]
                                       )

    ###Multi-GPU
    # overall_accuracy_sum = tf.Variable(tf.zeros(shape=([]), dtype=tf.float32),
    #                                    trainable=False,
    #                                    name="overall_accuracy_result",
    #                                    aggregation=tf.VariableAggregation.SUM,
    #                                    collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES]
    #                                    )

    update_op =  tf.compat.v1.assign_add(overall_accuracy_sum, overall_accuracy)

    return overall_accuracy, update_op


def summary(metric_op, loss_op):
    """
    minimal summaries for training @ monitoring
    """

    tf.compat.v1.summary.scalar("accuracy", metric_op)
    tf.compat.v1.summary.scalar("loss", loss_op)

    return tf.compat.v1.summary.merge_all()


def input(features, labels):

    with tf.name_scope("raw"):
        x250, x500, doy, year = features

        x250 = tf.cast(x250, tf.float32, name="x250")
        x500 = tf.cast(x500, tf.float32, name="x500")
        doy = tf.cast(doy, tf.float32, name="doy")
        year = tf.cast(year, tf.float32, name="year")

    y = tf.cast(labels, tf.int32, name="y")

    # integer sequence lenths per batch for dynamic_rnn masking
    sequence_lengths = tf.reduce_sum(tf.cast(x250[:, :, 0, 0, 0] > 0, tf.int32), axis=1,
                                                name="sequence_lengths")

    def resize(tensor, new_height, new_width):
        b = tf.shape(tensor)[0]
        t = tf.shape(tensor)[1]
        h = tf.shape(tensor)[2]
        w = tf.shape(tensor)[3]
        d = tf.shape(tensor)[4]

        # stack batch on times to fit 4D requirement of resize_tensor
        stacked_tensor = tf.reshape(tensor, [b * t, h, w, d])
        reshaped_stacked_tensor = tf.image.resize(stacked_tensor, size=(new_height, new_width))
        return tf.reshape(reshaped_stacked_tensor, [b, t, new_height, new_width, d])

    def expand3x(vector):
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        return vector

    with tf.name_scope("reshaped"):
        b = tf.shape(x250)[0]
        t = tf.shape(x250)[1]
        px = tf.shape(x250)[2]


        x500 = tf.identity(resize(x500, px, px), name="x500")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(x250, name="x250"))
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x500)

        # expand
        doymat = tf.multiply(expand3x(doy), tf.ones((b, t, px, px, 1)), name="doy")
        yearmat = tf.multiply(expand3x(year), tf.ones((b, t, px, px, 1)), name="year")

        x = tf.concat((x250, x500, doymat, yearmat), axis=-1, name="x")
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)

        if FLAGS.experiment == 'bands':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bandswodoy':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bands250m':
            bands250m = 2
            bands500m = 1
        elif FLAGS.experiment == 'bandswoblue':
            bands250m = 2
            bands500m = 4
        elif FLAGS.experiment == 'bandsaux':
            bands250m = 5
            bands500m = 5
        elif FLAGS.experiment == 'evi2':
            bands250m = 1
            bands500m = 1
        elif FLAGS.experiment == 'indices':
            bands250m = 7
            bands500m = 5
        else:
            bands250m = 10
            bands500m = 5

        # set depth of x for convolutions
        depth = bands250m + bands500m + 2  # doy and year
        # depth = bands250m + bands500m + 2  # doy and year

        # dynamic shapes. Fill for debugging
        x.set_shape([None, None, FLAGS.pix250m, FLAGS.pix250m, depth])
        y.set_shape((None, None, FLAGS.pix250m, FLAGS.pix250m))
        sequence_lengths.set_shape(None)

    return (x, sequence_lengths), (y,)


def input_eval(features):
    with tf.name_scope("raw"):
        x250, x500, doy, year = features

        x250 = tf.cast(x250, tf.float32, name="x250")
        x500 = tf.cast(x500, tf.float32, name="x500")
        doy = tf.cast(doy, tf.float32, name="doy")
        year = tf.cast(year, tf.float32, name="year")

    # integer sequence lengths per batch for dynamic_rnn masking
    sequence_lengths = tf.reduce_sum(tf.cast(x250[:, :, 0, 0, 0] > 0, tf.int32), axis=1,
                                                name="sequence_lengths")

    def resize(tensor, new_height, new_width):
        b = tf.shape(tensor)[0]
        t = tf.shape(tensor)[1]
        h = tf.shape(tensor)[2]
        w = tf.shape(tensor)[3]
        d = tf.shape(tensor)[4]

        # stack batch on times to fit 4D requirement of resize_tensor
        stacked_tensor = tf.reshape(tensor, [b * t, h, w, d])
        reshaped_stacked_tensor = tf.image.resize(stacked_tensor, size=(new_height, new_width))
        return tf.reshape(reshaped_stacked_tensor, [b, t, new_height, new_width, d])

    def expand3x(vector):
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        return vector

    with tf.name_scope("reshaped"):
        b = tf.shape(x250)[0]
        t = tf.shape(x250)[1]
        px = tf.shape(x250)[2]

        x500 = tf.identity(resize(x500, px, px), name="x500")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(x250, name="x250"))
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x500)

        # expand
        doymat = tf.multiply(expand3x(doy), tf.ones((b, t, px, px, 1)), name="doy")
        yearmat = tf.multiply(expand3x(year), tf.ones((b, t, px, px, 1)), name="year")

        x = tf.concat((x250, x500, doymat, yearmat), axis=-1, name="x")
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)

        if FLAGS.experiment == 'bands':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bandswodoy':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bands250m':
            bands250m = 2
            bands500m = 1
        elif FLAGS.experiment == 'bandswoblue':
            bands250m = 2
            bands500m = 4
        elif FLAGS.experiment == 'bandsaux':
            bands250m = 5
            bands500m = 5
        elif FLAGS.experiment == 'evi2':
            bands250m = 1
            bands500m = 1
        elif FLAGS.experiment == 'indices':
            bands250m = 7
            bands500m = 5
        else:
            bands250m = 10
            bands500m = 5

        FLAGS.num_bands_250m = bands250m
        FLAGS.num_bands_500m = bands500m

        # set depth of x for convolutions
        depth = FLAGS.num_bands_250m + FLAGS.num_bands_500m + 2  # doy and year
        # depth = bands250m + bands500m + 2  # doy and year

        # dynamic shapes. Fill for debugging
        x.set_shape([None, None, FLAGS.pix250m, FLAGS.pix250m, depth])
        sequence_lengths.set_shape(None)

    return x, sequence_lengths


def _model_fn(features, labels, mode, params):
  """MTLCC model

  Args:
    features: a batch of features.
    labels: a batch of labels or None if predicting.
    mode: an instance of tf.estimator.ModeKeys.
    params: a dict of additional params.

  Returns:
    A tf.estimator.EstimatorSpec that fully defines the model that will be run
      by an Estimator.
  """

  def parse(var, dtype):
      if type(var) == dtype:
          return var
      else:
          return eval(var)

  num_classes = parse(FLAGS.num_classes, int)
  print(params)

  if mode == tf.estimator.ModeKeys.PREDICT:
      # input pipeline
      with tf.name_scope("input"):
          x, sequence_lengths = input_eval(features)

      logits = inference(input=(x, sequence_lengths), num_classes=num_classes)

      prediction_scores = tf.nn.softmax(logits=logits, name="prediction_scores")
      pred = tf.argmax(prediction_scores, 3, name="predictions")

      predictions = {
          "pred": pred,
          "pred_sc": prediction_scores}

      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  else:
      with tf.name_scope("input"):
          (x, sequence_lengths), (alllabels,) = input(features, labels)
          alllabels = alllabels

      logits = inference(input=(x, sequence_lengths), num_classes=num_classes)

      # take first label -> assume labels do not change over timeseries
      first_labelmap = alllabels[ :, 0 ]

      # create one-hot labelmap from 0-num_classes
      labels = tf.one_hot(first_labelmap, num_classes + 1)

      # # mask out class 0 -> unknown
      unknown_mask = tf.cast(labels[ :, :, :, 0 ], tf.bool)
      not_unknown_mask = tf.logical_not(unknown_mask)

      # keep rest of labels
      labels = labels[ :, :, :, 1: ]

      global_step = tf.compat.v1.train.get_or_create_global_step()

      loss_op = loss(logits=logits, labels=labels, mask=not_unknown_mask, name="loss")
      # loss_op = softmax_focal_loss(logits=logits, labels=labels, mask=not_unknown_mask, gamma=2, alpha=0.25, name="loss")

      ao_ops = eval_oa(logits=logits, labels=labels, mask=not_unknown_mask)
      summary(ao_ops[0], loss_op)

      if mode == tf.estimator.ModeKeys.EVAL:
          eval_metric_ops = {"accuracy": ao_ops}

          # # Ask for accuracy and loss in each steps
          # class _CustomLoggingHook(tf.estimator.SessionRunHook):
          #     def begin(self):
          #         self.val_accuracy = [ ]
          #         self.val_loss = [ ]
          #         self.result_step = [ ]
          #         self.val_loss = [ ]
          #
          #     def before_run(self, run_context):
          #         return tf.train.SessionRunArgs([ao_ops[0], loss_op, global_step])
          #
          #     def after_run(self, run_context, run_values):
          #         result_accuracy, result_loss, result_step = run_values.results
          #         if result_step % 10 == 0:
          #             self.val_accuracy.append(result_accuracy)
          #             self.val_loss.append(result_loss)
          #         if result_step % int(params.save_checkpoints_steps) == 0:  # save logs in each 100 steps
          #             run.log_list('result_step', result_step)
          #             run.log_list('result_save_checkpoints_steps', int(params.save_checkpoints_steps))
          #             run.log_list('val_accuracy', self.val_accuracy)
          #             run.log_list('val_loss', self.val_loss)
          #             self.val_accuracy = [ ]
          #             self.val_loss = [ ]

          return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops=eval_metric_ops)
          # return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops=eval_metric_ops, evaluation_hooks = [_CustomLoggingHook()])

      elif mode == tf.estimator.ModeKeys.TRAIN:
          print("building optimizer...")
          train_op = optimize(loss_op, global_step, FLAGS.optimizertype, name="train_op")
          logging_hook = tf.estimator.LoggingTensorHook({"accuracy": ao_ops[0], 'loss': loss_op,  'global_step': global_step}, every_n_iter=64)

          # Ask for accuracy and loss in each steps
          class _CustomLoggingHook(tf.estimator.SessionRunHook):
              def begin(self):
                  self.training_accuracy = [ ]
                  self.training_loss = [ ]
                  self.training_step = [ ]

              def before_run(self, run_context):
                  return tf.train.SessionRunArgs([ao_ops[0], loss_op, global_step])

              def after_run(self, run_context, run_values):
                  result_accuracy, result_loss, result_step = run_values.results
                  if result_step % int(params.save_checkpoints_steps) == 0:
                      self.training_accuracy.append(result_accuracy)
                      self.training_loss.append(result_loss)
                      self.training_step.append(result_step)

                  if result_step % int(params.save_checkpoints_steps) == 0:  # save logs in each 100 steps
                      run.log_list('training_step', self.training_step)
                      run.log_list('training_accuracy', self.training_accuracy)
                      run.log_list('training_loss', self.training_loss)
                      self.training_accuracy = [ ]
                      self.training_loss = [ ]
                      self.training_step = [ ]

          # write FLAGS to file
          cfg = configparser.ConfigParser()
          flags_dict = dict()
          for name in FLAGS:
              flags_dict[name] = str(FLAGS[name].value)

          cfg["flags"] = flags_dict  # FLAGS.__flags #  broke tensorflow=1.5

          if not os.path.exists(params.model_dir):
              os.makedirs(params.model_dir)
          path = os.path.join(params.model_dir, MODEL_CFG_FILENAME)

          print("writing parameters to {}".format(path))
          with file_io.FileIO(path, 'w') as configfile:  # gcp
              cfg.write(configfile)

          # return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op, training_hooks=[logging_hook])
          return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op, training_hooks=[logging_hook, _CustomLoggingHook()])

      else:
          raise NotImplementedError('Unknown mode {}'.format(mode))


def convrnn_layer(input, filter, is_train=True, kernel=FLAGS.convrnn_kernel, sequence_lengths=None, bidirectional=True,
                  convcell='gru', scope="convrnn"):
    with tf.compat.v1.variable_scope(scope):

        x = input

        px = x.get_shape()[3]

        if FLAGS.convrnn_compression_filters > 0:
            x = conv_bn_relu(input=x, is_train=is_train, filter=FLAGS.convrnn_compression_filters, kernel=(1, 1, 1),
                             dilation_rate=(1, 1, 1), var_scope="comp")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(input, "input"))

        if convcell == 'gru':
            cell = ConvGRUCell((px, px), filter, kernel, activation=tf.nn.tanh,
                                       normalize=FLAGS.convrnn_normalize)

        elif convcell == 'lstm':
            cell = ConvLSTMCell((px, px), filter, kernel, activation=tf.nn.tanh,
                                        normalize=FLAGS.convrnn_normalize, peephole=FLAGS.peephole)
        else:
            raise ValueError("convcell argument {} not valid either 'gru' or 'lstm'".format(convcell))

        ## add dropout wrapper to cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=FLAGS.recurrent_dropout_i,
                                             state_keep_prob=FLAGS.recurrent_dropout_i)

        if bidirectional:
            outputs, final_states = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=x,
                                                                              sequence_length=sequence_lengths,
                                                                              dtype=tf.float32, time_major=False,
                                                                              swap_memory=FLAGS.swap_memory)

            concat_outputs = tf.concat(outputs, -1)

            if convcell == 'gru':
                concat_final_state = tf.concat(final_states, -1)
            elif convcell == 'lstm':
                fw_final, bw_final = final_states
                concat_final_state = LSTMStateTuple(
                    c=tf.concat((fw_final.c, bw_final.c), -1),
                    h=tf.concat((fw_final.h, bw_final.h), -1)
                )

        else:
            concat_outputs, concat_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=input,
                                                                   sequence_length=sequence_lengths,
                                                                   dtype=tf.float32, time_major=False)

        if convcell == 'lstm':
            concat_final_state = concat_final_state.c

        else:
            tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,
                                           tf.identity(concat_outputs, name="outputs"))
            tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,
                                           tf.identity(concat_final_state, name="final_states"))

        return concat_outputs, concat_final_state


def conv_bn_relu(var_scope="name_scope", is_train=True, **kwargs):
    with tf.compat.v1.variable_scope(var_scope):

        if FLAGS.cnn_activation == 'relu':
            activation_function = tf.nn.relu
        elif FLAGS.cnn_activation == 'leaky_relu':
            activation_function = tf.nn.leaky_relu
        else:
            raise ValueError("please provide valid 'cnn_activation' FLAG. either 'relu' or 'leaky_relu'")

        is_train = tf.constant(is_train, dtype=tf.bool)
        x = conv_layer(**kwargs)
        # x = Batch_Normalization(x, is_train, "bn") #deprecated replaced by keras API BN
        bn = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, moving_mean_initializer='zeros') #source https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/batch_normalization/mnist_classifier/trainer/model.py
        x = bn(x, training=is_train)
        x = activation_function(x)

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)
        return x


def conv_layer(input, filter, kernel, dilation_rate=(1, 1, 1), stride=1, conv_fun=tf.keras.layers.Conv3D,
               layer_name="conv"):  # based on https://github.com/tensorflow/tensorflow/issues/26145

    with tf.name_scope(layer_name):
        # pad input to required sizes for same output dimensions
        input = pad(input, kernel, dilation_rate, padding="REFLECT")

        network = conv_fun(use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='VALID',
                           dilation_rate=dilation_rate)(inputs=input)

        return network


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   # updates_collections=tf.GraphKeys.UPDATE_OPS,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def pad(input, kernel, dilation, padding="REFLECT"):
    """https://www.tensorflow.org/api_docs/python/tf/pad"""

    # determine required padding sizes
    def padsize(kernel, dilation):
        p = []
        for k, d in zip(kernel, dilation):
            p.append(int(int(k / 2) * d))
            # p.append(k/2)
        return p

    padsizes = padsize(kernel, dilation)

    # [bleft,bright], [tleft,tright], [hleft,hright], [wleft,wright],[dleft,dright]
    paddings = tf.constant([[0, 0]] + [[p, p] for p in padsizes] + [[0, 0]], dtype=tf.int32)

    return tf.pad(input, paddings, padding)


def build_estimator(run_config):
    """Returns TensorFlow estimator."""

    estimator = tf.estimator.Estimator(
        model_fn=_model_fn,
        config=run_config,
        params=run_config,
    )

    return estimator

#
# define utils functions
def convolution(inputs,W,data_format):
    """wrapper around tf.nn.convolution with custom padding"""
    pad_h = int(int(W.get_shape()[0])/2)
    pad_w = int(int(W.get_shape()[1])/2)

    paddings = tf.constant([[0, 0], [pad_h,pad_h], [pad_w,pad_w], [0, 0]],dtype=tf.int32)

    inputs_padded = tf.pad(inputs, paddings, "REFLECT")

    return tf.nn.convolution(inputs_padded, W, 'VALID', data_format=data_format)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeldir', type=str, default='./outputs', help="directory containing TF graph definition 'graph.meta'")
    parser.add_argument('--datadir', type=str, default='./data/24',
                        help='directory containing the data (defaults to environment variable $datadir)')
    parser.add_argument('-g', '--gpu', type=str, default="0", help='GPU')
    parser.add_argument('-d', '--train_on', type=str, default="2001 2002",
                        help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
    parser.add_argument('-e', '--epochs', type=int, default=5, help="epochs")
    parser.add_argument('-t', '--temporal_samples', type=int, default=None, help="Temporal subsampling of dataset. "
                                                                                 "Will at each get_next randomly choose "
                                                                                 "<temporal_samples> elements from "
                                                                                 "timestack. Defaults to None -> no temporal sampling")
    parser.add_argument('--save_frequency', type=int, default=64, help="save frequency")
    parser.add_argument('--summary_frequency', type=int, default=64, help="summary frequency")
    parser.add_argument('-f', '--fold', type=int, default=0, help="fold (requires train<fold>.ids)")
    parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
    parser.add_argument('--max_models_to_keep', type=int, default=5, help="maximum number of models to keep")

    parser.add_argument('--save_every_n_hours', type=int, default=1, help="save checkpoint every n hours")
    parser.add_argument('--queue_capacity', type=int, default=256, help="Capacity of queue")
    parser.add_argument('--allow_growth', type=bool, default=True, help="Allow dynamic VRAM growth of TF")
    parser.add_argument('--limit_batches', type=int, default=-1,
                        help="artificially reduce number of batches to encourage overfitting (for debugging)")
    parser.add_argument('-step', '--step', type=str, default="training", help='step')
    parser.add_argument('-experiment', '--experiment', type=str, default="bands", help='Experiment to train')
    parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6stable01to15_LCProp2_major",
                        help='Reference dataset to train')

    # args, _ = parser.parse_known_args()
    args, _ = parser.parse_known_args(args=argv[1:])
    return args

def train_and_evaluate(args):
    # clean checkpoint and model folder if exists
    if os.path.exists(args.modeldir) :
        for file_name in os.listdir(args.modeldir):
            file_path = os.path.join(args.modeldir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    """Runs model training and evaluation using TF Estimator API."""

    num_samples_train = 0
    num_samples_test = 0

    # if if num batches artificially reduced -> adapt sample size
    if args.limit_batches > 0:
        num_samples_train = args.limit_batches * args.batchsize * len(args.train_on.split(' '))
        num_samples_test = args.limit_batches * args.batchsize * len(args.train_on.split(' '))
    else:
        num_samples_train_out, _ = input_filenames(args, mode=tf.estimator.ModeKeys.TRAIN)
        num_samples_train += int(num_samples_train_out.get_shape()[0]) * len(args.train_on.split(' '))
        num_samples_test_out, _ = input_filenames(args, mode=tf.estimator.ModeKeys.EVAL)
        num_samples_test += len(num_samples_test_out) * len(args.train_on.split(' '))

    train_steps = num_samples_train / args.batchsize * args.epochs
    test_steps = num_samples_test / args.batchsize
    ckp_steps = num_samples_train / args.batchsize

    train_input_fn = functools.partial(
        input_fn_train_multiyear,
        args,
        mode=tf.estimator.ModeKeys.TRAIN
    )

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=train_steps
    )

    eval_input_fn = functools.partial(
        input_fn_train_multiyear,
        args,
        mode=tf.estimator.ModeKeys.EVAL
    )

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        start_delay_secs=0,
        throttle_secs=1,  # eval no more than every x seconds
        steps=test_steps, # evals on x batches        steps=test_steps, # evals on x batches
        name='eval'
    )

    session_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True,
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_steps=ckp_steps,
        save_summary_steps=ckp_steps,
        keep_checkpoint_max=args.max_models_to_keep,
        keep_checkpoint_every_n_hours=args.save_every_n_hours,
        model_dir=args.modeldir,
        log_step_count_steps=args.summary_frequency # set the frequency of logging steps for loss function
    )

    estimator = build_estimator(run_config)

    start_train_time = time.time()
    eval_res = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    train_time = round(time.time() - start_train_time, 2)

    print('Training time (s): ', train_time)

    # send logs to AML
    run.log('final_accuracy', eval_res[0]['accuracy'])
    run.log('final_loss', eval_res[0]['loss'])


def main():
  args = parse_arguments(sys.argv)
  train_and_evaluate(args)


if __name__ == "__main__":
  main()