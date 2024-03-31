
# Object Detection with TensorFlow Hub

This repository contains code for performing object detection using TensorFlow Hub with a pre-trained Faster R-CNN model based on Inception ResNet V2 architecture.

## Usage

1. Clone the repository:
2. Install the required dependencies: pip install tensorflow matplotlib tensorflow_hub
4. Run the object detection script:

```bash
python object_detection.py
```

## Description

The `object_detection.py` script loads a pre-trained Faster R-CNN model from TensorFlow Hub and performs object detection on an input image. The detected objects and their bounding boxes are then displayed.

## Dependencies

- TensorFlow
- TensorFlow Hub

## Pre-trained Model

The pre-trained model used in this code is available on TensorFlow Hub. You can find more details about the model [here]([https://tfhub.dev/your-pretrained-model](https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/frameworks/tensorFlow2/variations/640x640/versions/1/code)).

## Example

Here is an example of how to use the script:

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/your-pretrained-model")

# Read an image
img = tf.io.read_file('path/to/your/image.jpg')
img = tf.image.decode_jpeg(img, channels=3)
c_img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.expand_dims(c_img, axis=0)

# Perform object detection
results = detector(img.numpy())

print(results)
```
