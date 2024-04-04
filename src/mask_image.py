import numpy as np
from PIL import Image
import os

#image_path: path of main image, mask_image path of mask image, output_path: path to output result, 
#invert: invert the effect of mask (removes masked portion by default, set to true to only keep the masked portion)
def black_out_region_save(images_path, mask_path, output_path, invert = False):
    # Open images
    main_image = np.array(Image.open(images_path))
    mask_image = np.array(Image.open(images_path).convert('L'))

    # create the mask
    threshold = 128
    mask_image = ((mask_image > threshold) ^ invert)

    result_image = main_image.copy()

    # Set mask pixels to black
    result_image[mask_image] = [0, 0, 0]

    Image.fromarray(result_image).save(output_path)

#same but takes in np images instead and returns an np image instead of saving
def black_out_region_return(image, mask,  invert = False):
    # create the mask
    threshold = 128
    mask_image = ((mask_image > threshold) ^ invert)

    result_image = image.copy()

    # Set mask pixels to black
    result_image[mask_image] = [0, 0, 0]

    return result_image

def black_out_region_bulk(images_folder, mask_folder, output_folder, seperator = "_mask"):
    # Open images
    files = os.listdir(images_folder)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff'))]

    image_paths = [os.path.join(images_folder, image_file) for image_file in image_files]
    
    images = {os.path.basename(image_path).split('.')[0]: (np.array(Image.open(image_path)), image_path.split('.')[-1]) for image_path in image_paths}
    
    threshold = 128
    mask_files = os.listdir(mask_folder)
    mask_paths = [os.path.join(mask_folder, mask_file) for mask_file in mask_files]
    for mask_path in mask_paths: 
        mask_image = np.array(Image.open(mask_path).convert('L'))
        mask_image = mask_image < threshold
        image_name, mask_no = os.path.basename(mask_path).split(seperator)
        image, image_type = images[image_name]
        result_image = image.copy()
        result_image[mask_image] = [0,0,0]
        
        Image.fromarray(result_image).save(os.path.join(output_folder, image_name + mask_no + "." + image_type))


if __name__ == "__main__":
    images_folders = "./images"
    mask_folder = "./masks"
    output_folder = "./output"
    
    black_out_region_bulk(images_folder, mask_folder, output_folder)

