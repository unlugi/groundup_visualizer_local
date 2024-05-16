import cairosvg
import xml.etree.ElementTree as ET

ns = {'svg': "http://www.w3.org/2000/svg",
      'inkscape': "http://www.inkscape.org/namespaces/inkscape"}

def extract_group(svg_path, group_id, output_path):
    # Parse the SVG file
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Find the group with the specified ID
    target_group = None
    for group in root.findall('.//{http://www.w3.org/2000/svg}g'):
        if 'id' in group.attrib and group.attrib['id'] == group_id:
            target_group = group
            break

    if target_group is None:
        print(f"Group with ID '{group_id}' not found.")
        return

    # Create a new SVG tree with only the target group
    new_root = ET.Element('{http://www.w3.org/2000/svg}svg', attrib=root.attrib)
    new_root.append(target_group)

    # TODO: iterate through new_root to process paths and augment them
    for i, path in enumerate(new_root.findall('svg:g/svg:g/', ns)):
        path.attrib['stroke-width'] = "3.0"
        path.attrib['stroke'] = "rgb(0, 255, 0)"

    # Convert the new SVG tree to a string
    svg_content = ET.tostring(new_root, encoding='unicode')

    # Render the modified SVG content
    cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_path)

# Path to the SVG file
svg_path = 'sketch_group_tag.svg'

# Output file path for the rendered image
output_path = 'output3.png'

# ID of the group you want to render
group_id = 'ViewLayer_Lineset_Building_Roof_Topdown'
# group_id = "ViewLayer_Lineset_Building_Topdown"

# Extract and render the specific group from the SVG
extract_group(svg_path, group_id, output_path)