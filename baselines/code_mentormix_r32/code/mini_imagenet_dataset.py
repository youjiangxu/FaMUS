# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data for the Cifar100 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

_FILE_PATTERN = '%s/data'

_DATASET_DIR = ('')

_SPLITS_TO_SIZES = {
    'red_noise_nl_0.0': 50000,
    'red_noise_nl_0.05': 50000,
    'red_noise_nl_0.1': 50000,
    'red_noise_nl_0.15': 50000,
    'red_noise_nl_0.2': 50000,
    'red_noise_nl_0.3': 49985,
    'red_noise_nl_0.4': 50010,
    'red_noise_nl_0.5': 49962,
    'red_noise_nl_0.6': 50000,
    'red_noise_nl_0.8': 50008, 
    'clean_validation': 5000}

_NUM_CLASSES = 100

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'image/class/label': 'A single integer between 0 and 99.',
    'image/format': 'a string indicating the image format.',
    'image/class/fine_label': 'A single integer between 0 and 99.',
}


def get_split(split_name, dataset_dir=None):
  """Gets a dataset tuple with instructions for reading cifar100.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.

  Returns:
    A `Dataset` namedtuple. Image tensors are integers in [0, 255].

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
      'image/class/fine_label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  if split_name != 'clean_validation':
    items_to_handlers = {
        'image': tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': tfexample_decoder.Tensor('image/class/label'),
    }
  else:
    items_to_handlers = {
        'image': tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': tfexample_decoder.Tensor('image/class/fine_label'),
    }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
