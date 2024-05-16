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


def read_svg(path, change_thickness=False, width=256, height=256):
    """ Read a .svg file and return it as a numpy array. """
    # Read the sketch .svg
    tree = ET.parse(path)
    root = tree.getroot()

    if change_thickness:
        for i, path in enumerate(root.findall('svg:g/svg:g/', ns)):
            path.attrib['stroke-width'] = "1.0"

    # negate_colors changes stroke color
    img = svg2png(bytestring=ET.tostring(root), background_color='rgb(0,0,0)', negate_colors=True,
                  output_width=width, output_height=height)
    # image = Image.open(io.BytesIO(image))
    # Image._show(image)
    return np.array(Image.open(io.BytesIO(img)).convert('L'), dtype=np.float32)


# Read the PNG image
data_folder = "data/"
filename_sketch = "rgb_17500.png0001"
image_file_path = os.path.join(data_folder, "sketch/{}.svg".format(filename_sketch))

image = read_svg(image_file_path, change_thickness=True, width=1024, height=1024)

# Threshold the image to get a binary image
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

#[Next, Previous, First_Child, Parent]
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
final_contours = []

cc = 0

# TODO: check if we can get rid of the shrinking

#[Next, Previous, First_Child, Parent]
for i in range(len(contours)):
    current_children_stack = []
    current_contour_area_mask = np.zeros_like(image)
    if True: # hierarchy[0][i][3] != -1: # skip the root parent contour and get the first child
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
                current_child_index = hierarchy[0][current_child_index][0]
    if np.count_nonzero(current_contour_area_mask) > 0: # TODO: need to check if contour non-zero pixels < contourArea
        # if not (hierarchy[0][i][2]<0 and hierarchy[0][i][3]<0):

        non_zero_pixels = np.count_nonzero(current_contour_area_mask)
        if np.abs(non_zero_pixels - cv2.contourArea(contours[i])) < np.abs(non_zero_pixels - cv2.arcLength(contours[i], True)):
            contour_area_masks.append(current_contour_area_mask)
            final_contours.append(contours[i])
            cc += 1
            print(cc)


# Read depth
filename_depth = 'depth_17500.exr0001'
depth_file_path = os.path.join(data_folder, "depth/{}.exr".format(filename_depth))
depth_map = read_depth(depth_file_path, resize=True, img_size=(1024, 1024))
mean_depth = []
# For each mask, calculate the mean depth value
for mask in contour_area_masks:
    mean_depth.append(np.mean(depth_map[mask > 0]))

np.save('data/output/mean_depth{}.npy'.format(filename_sketch), mean_depth)
print(mean_depth)

# Display images
cv2.imshow('Parent Contours', parent_contour_image)
cv2.imshow('Child Contours', child_contour_image)

# for i, mask in enumerate(contour_area_masks):
#     cv2.imshow('Intermediate Region {} '.format(str(i)), mask)


# Convert each mask to a 3-channel image with different colors
colored_masks = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in contour_area_masks]

# Merge the colored masks into a single image
merged_mask = np.zeros_like(colored_masks[0], np.uint8)

for i, colored_mask in enumerate(colored_masks):
    merged_mask[colored_mask > 0] = (i+1)*10  # Assign different intensity for each mask

merged_mask = cv2.cvtColor(merged_mask, cv2.COLOR_BGR2RGB)

# Display the merged mask
cv2.imshow('Merged Masks', merged_mask)

# Going through every contour found in the image.
merged_mask_contours = np.zeros_like(merged_mask)
font = cv2.FONT_HERSHEY_COMPLEX
contour_paths = []

for cnt in final_contours:

    approx = cv2.approxPolyDP(curve=cnt, epsilon=0.005 * cv2.arcLength(cnt, True), closed=True)

    cv2.drawContours(merged_mask_contours, [approx], 0, (0, 0, 255), 1)

    # Used to flatten the array containing the co-ordinates of the vertices.
    n = approx.ravel()
    contour_paths.append(n)

    i = 0
    for j in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]

            # String containing the co-ordinates.
            string = str(x) + " " + str(y)

            if i == 0:
                # text on topmost co-ordinate.
                # cv2.putText(merged_mask_contours, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))
                cv2.circle(merged_mask_contours, (int(x), int(y)), color=(255, 0, 0), radius=2, thickness=-1)
            else:
                # text on remaining co-ordinates.
                # cv2.putText(merged_mask_contours, string, (x, y), font, 0.5, (0, 255, 0))
                cv2.circle(merged_mask_contours, (int(x), int(y)), color=(0, 255, 0), radius=2, thickness=-1)

        # break
        i = i + 1

#Save to SVG
# Generate SVG files for each path
image_size = (1024, 1024)
svg_content = '<?xml version="1.0" encoding="ascii"?>'
svg_content += '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{}" height="{}">'.format(str(image_size[0]), str(image_size[1]))
for path in contour_paths:
    svg_content += '\n <path d="M{} {} '.format(path[0], path[1])
    for i in range(2, len(path), 2):
        svg_content += 'L{} {} '.format(path[i], path[i+1])
    svg_content += 'Z" fill="none" stroke="black"/>'
svg_content += '\n</svg>'

# Save SVG file
with open('data/output/path_{}.svg'.format(filename_sketch), 'w') as f:
    f.write(svg_content)

# Convert SVG to PNG (optional)
svg2png(bytestring=svg_content.encode('utf-8'), write_to='data/output/path_{}.png'.format(filename_sketch))

print("SVG files saved successfully!")



# Showing the final image.
cv2.imshow('Masks with Contours', merged_mask_contours)


print('ff')
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: save contours, masks, mean depth values to npy or other format

# Find the coordinates of the contours
# https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/