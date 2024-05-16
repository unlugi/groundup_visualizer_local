import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cairosvg import svg2svg, svg2png
import xml.etree.cElementTree as ET
import re
import io
import PIL.Image as Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

ns = {'svg': "http://www.w3.org/2000/svg",
      'inkscape': "http://www.inkscape.org/namespaces/inkscape"}


def read_depth(filepath, resize=False, img_size=(256, 256)):
    depth = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = depth[..., 2]

    if resize:
        img = Image.fromarray(depth)  # convert to PIL image for resize
        img = img.resize((img_size[0], img_size[0]), Image.BILINEAR, ) # Image.LANCZOS
        depth = np.array(img)

    return depth


def read_svg(path, change_thickness=False):
    """ Read a .svg file and return it as a numpy array. """
    # Read the sketch .svg
    tree = ET.parse(path)
    root = tree.getroot()

    if change_thickness:
        for i, path in enumerate(root.findall('svg:g/svg:g/', ns)):
            path.attrib['stroke-width'] = "2.0"

    # negate_colors changes stroke color
    img = svg2png(bytestring=ET.tostring(root), background_color='rgb(0,0,0)', negate_colors=True, )
    # image = Image.open(io.BytesIO(image))
    # Image._show(image)
    return np.array(Image.open(io.BytesIO(img)).convert('L'), dtype=np.float32)


# Read the PNG image
data_folder = "data/"
image_file_path = os.path.join(data_folder, "sketch/0002.svg")

image = read_svg(image_file_path, change_thickness=True)

image[image>127] = 255


# Find contours on the binary image
# contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# # Draw contours on the original image
# contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
#
# # Display the original image and the image with contours
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Image with Contours')
# plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.show()


# Initialize lists to store parent and child contours
parent_contours = []
child_contours = []

# Iterate through contours and hierarchy
for i in range(len(contours)):
    # Check if contour has parent (inner contour)
    if hierarchy[0][i][3] != -1:
        child_contours.append(contours[i])
    else:
        parent_contours.append(contours[i])

# Draw parent contours
parent_contour_image = cv2.drawContours(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), parent_contours, -1, (0, 255, 0), 2)

# Draw child contours
child_contour_image = cv2.drawContours(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), child_contours, -1, (255, 0, 0), 2)

# # Display images
# cv2.imshow('Parent Contours', parent_contour_image)
# cv2.imshow('Child Contours', child_contour_image)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()

contour_area_masks = []
current_children_stack = []

#[Next, Previous, First_Child, Parent]
for i in range(len(contours)):
    current_children_stack = []
    current_contour_area_mask = np.zeros_like(image)
    if hierarchy[0][i][3] != -1: # skip the root parent contour and get the first child
        # Draw the current contour on the mask image (positive)
        current_contour_area_mask = cv2.drawContours(current_contour_area_mask, contours, i, (255, 255, 255), thickness=cv2.FILLED)

        # Get child
        current_child_index = hierarchy[0][i][2]
        current_child = contours[current_child_index]
        current_children_stack.append(current_child)
        # process children
        while current_children_stack:
            # empty child region
            current_contour_area_mask = cv2.drawContours(current_contour_area_mask, contours, current_child_index, (0, 0, 0), thickness=cv2.FILLED)

            if hierarchy[0][current_child_index][0] == -1:
                current_children_stack.pop()
            else:
                current_children_stack.append(contours[hierarchy[0][current_child_index][0]])
    if np.count_nonzero(current_contour_area_mask) > 0: # TODO: need to check if contour non-zero pixels < contourArea
        # if not (hierarchy[0][i][2]<0 and hierarchy[0][i][3]<0):

        non_zero_pixels = np.count_nonzero(current_contour_area_mask)
        if np.abs(non_zero_pixels - cv2.contourArea(contours[i])) < np.abs(non_zero_pixels - cv2.arcLength(contours[i], True)):
            contour_area_masks.append(current_contour_area_mask)

        # break

# Read depth
depth_file_path = os.path.join(data_folder, "depth/Image0002.exr")
depth_map = read_depth(depth_file_path, resize=False)
mean_depth = []
# For each mask, calculate the mean depth value
for mask in contour_area_masks:
    mean_depth.append(np.mean(depth_map[mask > 0]))

print(mean_depth)


# Display images
cv2.imshow('Parent Contours', parent_contour_image)
cv2.imshow('Child Contours', child_contour_image)

for i, mask in enumerate(contour_area_masks):
    cv2.imshow('Intermediate Region {} '.format(str(i)), mask)


cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: save contours, masks, mean depth values to npy or other format

