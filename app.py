import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sampler import sample
from plotter import plot_gradcam_grid
from plotter_counter_style import create_contour_overlay

path = "test_data.txt"
model = tf.keras.models.load_model('model.keras')
last_conv_layer = 'Conv_1'  # final conv layer

def get_image_paths(new=False):
    images, labels = [], []
    if new:
        sample("images", 6, path)
    with open(path, 'r') as file:
        for line in file:
            img_path, label = line.strip().split(" ")
            images.append(img_path)
            labels.append(label)
    return images, labels

def load_image(path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0, img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_heatmap_visuals(img, heatmap, alpha=0.4):
    img = np.array(img).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_color[:, :, ::-1], overlayed[:, :, ::-1]  # Convert to RGB

# Main flow
image_paths, actual_labels = get_image_paths()
test_images = [load_image(p) for p in image_paths]

originals, heatmap_only_imgs, overlaid_imgs = [], [], []

for img_array, original_img in test_images:
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    heatmap_img, overlay_img = get_heatmap_visuals(original_img, heatmap)
    contour_img = create_contour_overlay(original_img, heatmap)

    originals.append(original_img)
    heatmap_only_imgs.append(heatmap_img)
    overlaid_imgs.append(contour_img)

plot_gradcam_grid(originals, heatmap_only_imgs, overlaid_imgs, save_path='gradcam_grid_output.png')
