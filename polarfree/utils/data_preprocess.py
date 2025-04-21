# import os
# import cv2
# import glob
# import numpy as np
# from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
# from tqdm import tqdm

# # Set input and output paths
# datapath = r'/ailab/user/yaomingde/workspace/ideas/polarization_reflect/data/v1/polar_origin'
# savepath = r'/ailab/user/yaomingde/workspace/ideas/polarization_reflect/data/v1/polar_rgb'

# def get_rgb(rgb):
#     rgb = rgb.astype(np.uint8)
#     rgb_demosaiced = demosaicing_CFA_Bayer_Malvar2004(rgb, "RGGB")
#     rgb_demosaiced = rgb_demosaiced[:,:,::-1]
#     return rgb_demosaiced


# # Process image function
# def process(img_path):
#     # Read the input image in grayscale
#     rgb = cv2.imread(img_path, 0).astype(float)
#     i0,i45,i90,i135 = rgb[0::2,0::2],rgb[0::2,1::2],rgb[1::2,0::2],rgb[1::2,1::2] 
#     i0_rgb = get_rgb(i0)
#     i45_rgb = get_rgb(i45)
#     i90_rgb = get_rgb(i90)
#     i135_rgb = get_rgb(i135)

#     # # Downscale the image by averaging blocks of 2x2 pixels
#     # rgb = (rgb[0::2, 0::2] + rgb[0::2, 1::2] + rgb[1::2, 0::2] + rgb[1::2, 1::2]) / 4.0
    
#     # # Convert the resulting image back to uint8
#     # rgb = rgb.astype(np.uint8)
    
#     # # Demosaicing using Malvar2004 method for "RGGB" pattern
#     # rgb_demosaiced = demosaicing_CFA_Bayer_Malvar2004(rgb, "RGGB")
#     # rgb_demosaiced = rgb_demosaiced[:,:,::-1]
    
#     return rgb_demosaiced

# # Get all images matching the file pattern
# imgs = glob.glob(os.path.join(datapath, '*', '*', '*.png'))

# # Process each image
# for img_path in tqdm(imgs):
#     # Process the image
#     out_img = process(img_path)
    
#     # Recreate the output path by replacing the root directory to match the original structure
#     relative_path = os.path.relpath(img_path, datapath)  # Get relative path to maintain structure
#     save_img_path = os.path.join(savepath, relative_path) # Append to save path
    
#     # Ensure the directories exist
#     os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    
#     # Save the processed image
#     cv2.imwrite(save_img_path, out_img)

# print("Processing complete!")
















import os
import cv2
import glob
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from tqdm import tqdm

# Set input and output paths
datapath = r'/ailab/user/yaomingde/workspace/ideas/polarization_reflect/data/v1/polar_origin'
savepath = r'/ailab/user/yaomingde/workspace/ideas/polarization_reflect/data/v1/polar_rgb'

def get_rgb(rgb):
    rgb = rgb.astype(np.uint8)
    rgb_demosaiced = demosaicing_CFA_Bayer_Malvar2004(rgb, "RGGB")
    rgb_demosaiced = rgb_demosaiced[:,:,::-1]
    return rgb_demosaiced


# Process image function
def process(img_path):
    # Read the input image in grayscale
    rgb = cv2.imread(img_path, 0).astype(float)
    i0,i45,i90,i135 = rgb[0::2,0::2],rgb[0::2,1::2],rgb[1::2,0::2],rgb[1::2,1::2] 
    i0_rgb = get_rgb(i0)
    i45_rgb = get_rgb(i45)
    i90_rgb = get_rgb(i90)
    i135_rgb = get_rgb(i135)

    # # Downscale the image by averaging blocks of 2x2 pixels
    # rgb = (rgb[0::2, 0::2] + rgb[0::2, 1::2] + rgb[1::2, 0::2] + rgb[1::2, 1::2]) / 4.0
    
    # # Convert the resulting image back to uint8
    # rgb = rgb.astype(np.uint8)
    
    # # Demosaicing using Malvar2004 method for "RGGB" pattern
    # rgb_demosaiced = demosaicing_CFA_Bayer_Malvar2004(rgb, "RGGB")
    # rgb_demosaiced = rgb_demosaiced[:,:,::-1]
    
    return i0_rgb,i45_rgb, i90_rgb , i135_rgb

# Get all images matching the file pattern
imgs = glob.glob(os.path.join(datapath, '*', '*', '*.png'))

# Process each image
for img_path in tqdm(imgs):
    # Process the image
    i0_rgb,i45_rgb, i90_rgb , i135_rgb = process(img_path)
    
    # Recreate the output path by replacing the root directory to match the original structure
    relative_path = os.path.relpath(img_path, datapath)  # Get relative path to maintain structure
    save_img_path = os.path.join(savepath, relative_path) # Append to save path
    
    save_img_path000 = save_img_path[:-4]+'_000.png'
    save_img_path045 = save_img_path[:-4]+'_045.png'
    save_img_path090 = save_img_path[:-4]+'_090.png'
    save_img_path135 = save_img_path[:-4]+'_135.png'
    # print(save_img_path135)
    # # Ensure the directories exist
    os.makedirs(os.path.dirname(save_img_path000), exist_ok=True)
    os.makedirs(os.path.dirname(save_img_path045), exist_ok=True)
    os.makedirs(os.path.dirname(save_img_path090), exist_ok=True)
    os.makedirs(os.path.dirname(save_img_path135), exist_ok=True)
    
    # # Save the processed image
    cv2.imwrite(save_img_path000, i0_rgb)
    cv2.imwrite(save_img_path045, i45_rgb)
    cv2.imwrite(save_img_path090, i45_rgb )
    cv2.imwrite(save_img_path135, i135_rgb)

print("Processing complete!")
