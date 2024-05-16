import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cairosvg import svg2svg, svg2png
import xml.etree.cElementTree as ET
import re
import io
import PIL.Image as Image
from typing import List, Tuple
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

ns = {'svg': "http://www.w3.org/2000/svg",
      'inkscape': "http://www.inkscape.org/namespaces/inkscape"}

# TODO: what I need is outer contour and child outer contours:
# (oc_parent, [oc_child1, oc_child2, ...])

# TODO: for buildings on the border, add a white border to avoid missing contours


class ContourExtractor:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.images = {}

    def read_depth(self, filepath: str, resize: bool = False, img_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        depth = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[..., 2]

        if resize:
            img = Image.fromarray(depth)  # convert to PIL image for resize
            img = img.resize((img_size[0], img_size[0]), Image.BILINEAR, ) # Image.LANCZOS
            depth = np.array(img)

        return depth

    def read_svg(self, path, change_thickness=False, width=256, height=256, add_border=False):
        tree = ET.parse(path)
        root = tree.getroot()

        if change_thickness:
            for i, path in enumerate(root.findall('svg:g/svg:g/', ns)):
                path.attrib['stroke-width'] = "1.5"
                path.attrib['stroke'] = 'black'  # Border color

        if add_border:
            # Create a rectangle element for the border
            border_rect = ET.Element('rect')
            border_rect.attrib['x'] = '0'
            border_rect.attrib['y'] = '0'
            border_rect.attrib['width'] = str(width -  2 )
            border_rect.attrib['height'] = str(height - 2 )
            border_rect.attrib['stroke'] = 'black'  # Border color
            border_rect.attrib['stroke-width'] = "1.5"  # Border width
            border_rect.attrib['fill'] = 'none'  # No fill

            # Add the rectangle to the root
            root.append(border_rect)

        img = svg2png(bytestring=ET.tostring(root), background_color='rgb(0,0,0)', negate_colors=True,
                      output_width=width, output_height=height)

        # Add border
        new_image = np.array(Image.open(io.BytesIO(img)).convert('L'), dtype=np.float32)

        new_image[0,...] = 255
        new_image[-1,...] = 255
        new_image[...,0] = 255
        new_image[...,-1] = 255

        return new_image

        # return np.array(Image.open(io.BytesIO(img)).convert('L'), dtype=np.float32)

    def extract_contours_from_sketch(self, filename_sketch: str) -> List[float]:
        image_file_path = os.path.join(self.data_folder, "sketch/{}.svg".format(filename_sketch))
        image = self.read_svg(image_file_path, change_thickness=True, width=1024, height=1024, add_border=False)
        image[image > 127] = 255

        self.images['sketch'] = image

        contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]

        contour_area_masks = []
        final_contours = []

        #[Next, Previous, First_Child, Parent]
        for i in range(len(contours)):
            current_children_stack = []
            current_contour_area_mask = np.zeros_like(image)
            if True:
            # if hierarchy[i][0] == -1 and hierarchy[i][1] == -1 and hierarchy[i][3] == -1:
            #     continue
            # else:
                current_contour_area_mask = cv2.drawContours(current_contour_area_mask, contours, i, (255, 255, 255), thickness=cv2.FILLED)
                current_child_index = hierarchy[i][2]
                current_child = contours[current_child_index]
                current_children_stack.append(current_child)
                while current_children_stack:
                    current_contour_area_mask = cv2.drawContours(current_contour_area_mask, contours, current_child_index, (0, 0, 0), thickness=cv2.FILLED)
                    if hierarchy[current_child_index][0] == -1:
                        current_children_stack.pop()
                    else:
                        current_children_stack.append(contours[hierarchy[current_child_index][0]])
                        current_child_index = hierarchy[current_child_index][0]
                if np.count_nonzero(current_contour_area_mask) > 0:
                    contour_area_masks.append(current_contour_area_mask)
                    final_contours.append(contours[i])

        return final_contours, contour_area_masks, hierarchy

    def get_parent_child_contours(self, contours, hierarchy):
        # Initialize lists to store parent and child contours
        parent_contours = []
        child_contours = []

        # [Next, Previous, First_Child, Parent]
        # Iterate through contours and hierarchy
        for i in range(len(contours)):
            # Check if contour has parent (inner contour)
            if hierarchy[i][3] != -1:
                child_contours.append(contours[i])
            else:
                parent_contours.append(contours[i])

        # Draw parent contours
        parent_image = cv2.drawContours(cv2.cvtColor(self.images['sketch'], cv2.COLOR_GRAY2BGR),
                                        parent_contours,
                                        -1,
                                        (0, 255, 0), 2)

        # Draw child contours
        child_image = cv2.drawContours(cv2.cvtColor(self.images['sketch'], cv2.COLOR_GRAY2BGR),
                                       child_contours,
                                       -1,
                                       (255, 0, 0), 2)

        self.images['parent_contours'] = parent_image
        self.images['child_contours'] = child_image



    def calculate_mean_depths_per_contour(self, contour_area_masks, filename_depth, filename_sketch):

        depth_file_path = os.path.join(self.data_folder, "depth/{}.exr".format(filename_depth))
        depth_map = self.read_depth(depth_file_path, resize=True, img_size=(1024, 1024))
        mean_depth = []

        for mask in contour_area_masks:
            mean_depth.append(np.mean(depth_map[mask > 0]))

        np.save('{}/output/mean_depth{}.npy'.format(self.data_folder, filename_sketch), mean_depth)

        return mean_depth

    def find_contour_coordinates(self, contour_area_masks, final_contours):

        # Convert each mask to a 3-channel image with different colors
        colored_masks = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in contour_area_masks]

        # Merge the colored masks into a single image
        merged_mask = np.zeros_like(colored_masks[0], np.uint8)

        for i, colored_mask in enumerate(colored_masks):
            merged_mask[colored_mask > 0] = (i + 1) * 10  # Assign different intensity for each mask

        merged_mask = cv2.cvtColor(merged_mask, cv2.COLOR_BGR2RGB)
        self.images['merged_mask'] = merged_mask

        # Going through every contour found in the image.
        merged_mask_contours = np.zeros_like(merged_mask)
        font = cv2.FONT_HERSHEY_COMPLEX
        contour_paths = []

        for cnt in final_contours:

            approx = cv2.approxPolyDP(curve=cnt, epsilon=0.001 * cv2.arcLength(cnt, True), closed=True)

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
        self.images['merged_mask_contours'] = merged_mask_contours

        return contour_paths


    def save_svg_contours(self, filename_sketch: str, contour_paths: List[np.ndarray], image_size: Tuple[int, int]) -> None:
        # Save to SVG
        # Generate SVG files for each path
        image_size = (1024, 1024)
        svg_content = '<?xml version="1.0" encoding="ascii"?>'
        svg_content += '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{}" height="{}">'.format(
            str(image_size[0]), str(image_size[1]))
        for path in contour_paths:
            svg_content += '\n <path d="M{} {} '.format(path[0], path[1])
            for i in range(2, len(path), 2):
                svg_content += 'L{} {} '.format(path[i], path[i + 1])
            svg_content += 'Z" fill="none" stroke="black"/>'
        svg_content += '\n</svg>'

        # Save SVG file
        with open('{}/output/path_{}.svg'.format(self.data_folder,filename_sketch), 'w') as f:
            f.write(svg_content)

        # Convert SVG to PNG (optional)
        svg2png(bytestring=svg_content.encode('utf-8'), write_to='{}/output/path_{}.png'.format(self.data_folder,filename_sketch))

        print("SVG files saved successfully!")

    def display_images(self, im_size):

        for key, value in self.images.items():
            fig, ax = plt.subplots(figsize=im_size, dpi=100)
            ax.imshow(value)
            ax.text(80, 10, key, fontsize=10, ha='center', va='center', color='white')  # Add text inside the image
            ax.axis('off')  # Turn off axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters
            plt.show()


if __name__ == '__main__':
    path_to_data = "data/"
    image_id = "0000" #'10001'
    fname_sketch = "rgb_{}.png0001".format(image_id)
    fname_depth = 'depth_{}.exr0001'.format(image_id)
    image_size = (1024, 1024)

    # Initialize the ContourExtractor class
    extractor = ContourExtractor(path_to_data)

    # Extract contours from the sketch
    contours, contour_region_masks, hierarchy = extractor.extract_contours_from_sketch(fname_sketch)

    # Calculate mean depth values per contour
    mean_depth_values = extractor.calculate_mean_depths_per_contour(contour_region_masks, fname_depth, fname_sketch)

    # Find contour coordinates by fitting polynomial curves and save the curves to SVG
    contour_paths_for_svg = extractor.find_contour_coordinates(contour_region_masks, contours)
    extractor.save_svg_contours(fname_sketch, contour_paths_for_svg, image_size)

    # Display images
    extractor.get_parent_child_contours(contours, hierarchy)
    extractor.display_images(im_size=(10.24, 10.24))

    print('Done!')
