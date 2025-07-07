import os.path
from torchvision import models, transforms
import torch
import cv2
import numpy as np
from PIL import Image
import random
import time

from process_whale_photo import whale_back_removal


def paste_whale(background_path, whale_path, output_path, output_path_bbox, annotation_path, fig_whale_size, attempts=10, photoshop_whale=False, with_cover=True, with_ripple=True, with_bbox=True, specialCare=False):
    """
        Pastes a segmented whale object onto sea region in a specified background image
        while maintaining natural realism through various augmentations.
        The function saves the resulting composite image (with or without bounding boxes) and a corresponding
        YOLO annotation file.

        Parameters:
        - background_path: Path to the background image where the whale will be placed.
        - whale_path: Path to the whale object image (without background).
        - output_path: Path to save the resulting image without bounding boxes.
        - output_path_bbox: Path to save the resulting image with bounding boxes (if enabled).
        - annotation_path: Path to save the YOLO annotation file for the whale.
        - fig_whale_size: Desired size of the whale figure as a tuple (width, height).
        - attempts: Number of placement attempts to find a suitable position (default: 10).
        - photoshop_whale: Flag to enable/disable standard augmentations (default: False).
        - with_cover: Whether to apply a sea texture blend to the whale's bottom region (default: True).
        - with_ripple: Whether to apply ripple effects to the sea region below the whale (default: True).
        - with_bbox: Whether to save an output image with bounding boxes (default: True).
        - specialCare: Flag to apply special handling , skipping ripple effects (default: False).

        Workflow:
        1. Load the background and whale images.
        2. Resize the whale image to the desired size.
        3. Attempt to place the whale on the background while avoiding overlap and ensuring realism.
        4. Apply augmentations, such as:
           - Adjusting whale colors, applying blur and blending with the sea texture for natural appearance.
           - Optional adding ripple effects in the sea region below the whale.
           - Optional rotations, flips, and noise
        5. Save the final composite image (optional saving additional one with bounding boxes).
        6. Generate a YOLO annotation file for the whale's position and size.

        Outputs:
        - Saves the composite image to `output_path`.
        - Optionally saves an additional image with bounding boxes to `output_path_bbox`.
        - Writes YOLO annotation data to `annotation_path`.
        """
    
    background_image = cv2.imread(background_path)
    #background_image = cv2.resize(background_image, (512, 512))
    whale_image = cv2.imread(whale_path,cv2.IMREAD_UNCHANGED)


    def adjust_whale_size(whale_image_2_adj, fig_whale_size_2_adj):
        """
        Adjust the size of the whale object based on the aspect ratio,
        ensuring the width and height do not exceed the original figure size.

        Parameters:
        - whale_image: The input whale image with an alpha channel (RGBA format).
        - fig_whale_size: A tuple (max_width, max_height) representing the maximum size for scaling.

        Returns:
        - adjusted_size: A tuple (new_width, new_height) for the resized whale image.
        """
        # Extract the alpha channel to find the bounding box
        alpha_channel = whale_image_2_adj[:, :, 3]
        non_transparent_pixels = np.where(alpha_channel > 40)

        # Determine the bounding box dimensions of the whale
        object_width = non_transparent_pixels[1].max() - non_transparent_pixels[1].min()
        object_height = non_transparent_pixels[0].max() - non_transparent_pixels[0].min()

        # Calculate the aspect ratio
        aspect_ratio = object_width / object_height

        # Max allowed size
        max_width, max_height = fig_whale_size_2_adj

        # Adjust the size based on the aspect ratio
        if aspect_ratio > 1:  # Wide object
            # Scale width to max_width and adjust height proportionally
            new_width = min(object_width, max_width)
            new_height = min(int(new_width / aspect_ratio), object_height)
        else:  # Tall object
            # Scale height to max_height and adjust width proportionally
            new_height = min(object_height, max_height)
            new_width = min(int(new_height * aspect_ratio), object_width)

        return (new_width, new_height)

    #Adujst whale figure size according to the sizes of the non-transpernt pixels
    fig_whale_size = adjust_whale_size(whale_image,fig_whale_size)
    whale_image = cv2.resize(whale_image, fig_whale_size)
    avg_color =None

    #Finding the horizon line and sea area in the background photo

    def flood_fill_operation(background_image, squra_len, atmps,tolerance=5):
        #finding seed point and radius
        bck_h, bck_w = background_image.shape[:2]
        top_h = int(bck_h*3/4)
        flag = False
        for i in range(atmps):
            x_rnd = random.randint(0, bck_w - squra_len)
            y_rnd = random.randint(top_h, bck_h - squra_len)
            # Extract pixel values in the circular region
            # Crop a square region around the seed point (bounding box of the circle)
            x_min = max(x_rnd - squra_len, 0)
            x_max = min(x_rnd + squra_len, bck_w - 1)
            y_min = max(y_rnd - squra_len, 0)
            y_max = min(y_rnd + squra_len, bck_h - 1)
            cropped_region = background_image[y_min:y_max, x_min:x_max]
            avg_color_temp = np.mean(cropped_region, axis=(0,1))
            b, g, r = avg_color_temp
            if b > r and g > r:
                flag=True
                break
        if not flag:
            print(f"Failed to find a suitable seed point in {atmps} attempts.")
            return None
        mask = np.zeros((bck_h + 2, bck_w + 2), np.uint8)
        # Compute the color range (min and max for each channel)
        lo_diff = [(max(0, b - tolerance)), (max(0, g - tolerance)), (max(0, r - tolerance))]
        up_diff = [(min(255, b + tolerance)), (min(255, g + tolerance)), (min(255, r + tolerance))]

        flood_fill_result = background_image.copy()
        cv2.floodFill(
            flood_fill_result,
            mask,
            seedPoint=(x_rnd, y_rnd),
            newVal=(0, 255, 0),  # Temporary color for debugging (green)
            loDiff=lo_diff,
            upDiff=up_diff,
            flags=cv2.FLOODFILL_FIXED_RANGE
        )

        # Remove the border padding from the mask
        flooded_area = mask[1:-1, 1:-1]
        # Return the flooded mask
        return flood_fill_result,flooded_area

    def detect_lines(background_image,canny_low_ts,canny_high_ts):

        # Detect horizon line
        gray_bg = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        edges = cv2.Canny(gray_bg, canny_low_ts, canny_high_ts) # setting lower and upper thresholds for detecting edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=50) # detect straight lines in the photo
        return lines

    def detect_water_region(image, threshold=30,atmps=10,squra_len=10):

        #find edges
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

        # finding seed point
        h, w = background_image.shape[:2]
        top_h = int(h * 3 / 4)
        flag = False
        for i in range(atmps):
            seed_x = random.randint(0, w - squra_len)
            seed_y = random.randint(top_h, h - squra_len)
            # Extract pixel values in the circular region
            # Crop a square region around the seed point (bounding box of the circle)
            x_min = max(seed_x - squra_len, 0)
            x_max = min(seed_x + squra_len, w - 1)
            y_min = max(seed_y - squra_len, 0)
            y_max = min(seed_y + squra_len, h - 1)
            cropped_region = background_image[y_min:y_max, x_min:x_max]
            avg_color_temp = np.mean(cropped_region, axis=(0, 1))
            b, g, r = avg_color_temp
            if b > r and g > r:
                flag = True
                break
        if not flag:
            print(f"Failed to find a suitable seed point in {atmps} attempts.")
            return None

        water_mask= np.zeros((h, w), np.uint8)  # To store the grown region
        seed_color = image[seed_y, seed_x]  # Color at the seed point
        # Stack for iterative region growing
        stack = [(seed_x, seed_y)]

        while stack:
            x, y = stack.pop()

            # Skip if out of bounds or already visited
            if x <  1 or x >= w-1 or y < 1 or y >= h-1 or water_mask[y, x] == 255:
                continue

            # Skip if it's an edge
            if edges[y, x] == 255:
                continue

            # Check color similarity
            pixel_color = image[y, x]
            diff = np.abs(pixel_color - seed_color).mean()
            if diff < threshold:
                # Add to the region
                water_mask[y, x] = 255

                # Add neighbors to the stack
                stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

        result = image.copy()
        result[water_mask == 255] = (0, 255, 0)  # Highlight the water region in green

        return result, water_mask

    def deeplab_detect_water_region(background_path_):

        # Load DeeplabV3 pretrained on COCO
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
        model.eval()

        original_image  = Image.open(background_path_).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(520),  # Resize to model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(original_image).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)["out"]

        # Postprocess the model output
        def postprocess_output(output, original_image):
            output_predictions = output.argmax(1).squeeze(0).detach().cpu().numpy()
            water_class_id = 13  # Check DeeplabV3 class mapping for "water"

            # Create a mask for the water region
            water_mask = (output_predictions == water_class_id).astype(np.uint8)

            # Apply the mask to the original image
            original_image = original_image.resize((output_predictions.shape[1], output_predictions.shape[0]))
            original_array = np.array(original_image)
            water_region = np.zeros_like(original_array)
            for i in range(3):  # Apply the mask to each color channel
                water_region[..., i] = original_array[..., i] * water_mask

            return water_region, water_mask

        water_region, water_mask = postprocess_output(output, original_image)

        # Save the image with the water mask applied
        water_image = Image.fromarray(water_region)

        return water_image,water_mask

    # Find the longest horizontal line
    horizon_y = None
    cur_att = 0
    horizontal_lines = []
    canny_l_ts = 50
    canny_h_ts = 150

    while(len(horizontal_lines) == 0 and cur_att<=1):
        lines = detect_lines(background_image, canny_l_ts, canny_h_ts)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) <30 and ((y2+y1)//2 < (background_image.shape[0]*2/3)) :  # ∆y
                    horizontal_lines.append(line)
        canny_l_ts = 30
        canny_h_ts =80
        cur_att+=1

    # save background photo with lines for debug - delete in the end!
    # lines_path = output_path_bbox[:-4] + "_lines" + output_path_bbox[-4:]
    # save_background_lines(background_image, lines, lines_path)

    #flood fill check
    # temp_bk_img,sea_region_mask = flood_fill_operation(background_image,5,10)
    # flood_path = final_image_path_bbox[:-4] + "_flood_chk" + final_image_path_bbox[-4:]
    # cv2.imwrite(flood_path, temp_bk_img)
    # print(f"Saved image with lines: {flood_path}")
    # return

    #Edge Detection + Region Growing check
    # temp_bk_img,sea_region_mask = detect_water_region(background_image)
    # temp_path = final_image_path_bbox[:-4] + "_edge_chk" + final_image_path_bbox[-4:]
    # cv2.imwrite(temp_path, temp_bk_img)
    # print(f"Saved image with lines: {temp_path}")
    # return

    #Deeplabv3 check
    # temp_bk_img,sea_region_mask = deeplab_detect_water_region(background_path)
    # temp_path = final_image_path_bbox[:-4] + "_deplab_chk" + final_image_path_bbox[-4:]
    # temp_bk_img.save(temp_path)
    # print(f"Saved image with lines: {temp_path}")
    # return

    # Detect the horizon line
    if len(horizontal_lines) != 0:
        longest_line = max(horizontal_lines, key=lambda line: line[0][2] - line[0][0])
        x1, y1, x2, y2 = longest_line[0]
        horizon_y =(y1 + y2) // 2


    # Finding position in the background photo for the whale to be paste on (only if horizon line detected)

    if horizon_y is not None:
        whale_h, whale_w = whale_image.shape[:2]
        bg_h, bg_w = background_image.shape[:2]

        # vertical boundaries: half bottom of the area between horizon line and bottom of the bckgrnd
        vertical_min =horizon_y+ (bg_h-horizon_y)//2
        vertical_max = bg_h - whale_h

        def is_too_many_lines(lines, image_shape):
            """
            Check if there are too many lines covering a large portion of the image.

            Parameters:
                lines: Detected lines from cv2.HoughLinesP.
                image_shape: Shape of the background image.

            Returns:
                bool: True if the lines cover too much of the image.
            """
            height, width = image_shape[:2]
            line_coverage_mask = np.zeros((height, width), dtype=np.uint8)

            # Draw all lines on the mask
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_coverage_mask, (x1, y1), (x2, y2), 255, thickness=1)

            # Calculate the percentage of pixels covered by lines
            covered_area_ratio = np.sum(line_coverage_mask > 0) / (height * width)
            return covered_area_ratio > 0.2  # Adjust the threshold as needed

        def filter_non_horizontal_lines(lines):
            """
            Filter out horizontal lines to reduce noise from waves.

            Parameters:
                lines: Detected lines from cv2.HoughLinesP.

            Returns:
                list: Filtered lines with non-horizontal orientation.
            """
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 10 and angle < 170:  # Exclude lines close to horizontal
                    filtered_lines.append(line)
            return filtered_lines

        # Check if the current set of lines is too dense
        if is_too_many_lines(lines, background_image.shape):
            # If too dense, filter out horizontal lines and try again
            lines = filter_non_horizontal_lines(lines)

        # checking for overlap whale figure on lines
        def check_no_overlap_and_color(x_offset, y_offset,cur_att):
            # Check if the area where we want to paste the whale contains a line
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                # Check if the y_offset to y_offset + whale_h overlaps with any detected line
                if (y_offset < y2_line and y_offset + whale_h > y1_line and
                    x_offset < x2_line and x_offset + whale_w > x1_line):
                    return False,None  # There is overlap
            # Check if the area where we want to paste the whale fits by color
            def is_sea_region(background_image, x_offset, y_offset, whale_w, whale_h, sea_hue_range=(90, 140),min_sea_ratio=0.5):
                """
                Check if the region is likely to be sea based on color matching.

                Parameters:
                    background_image (np.array): The background image (BGR format).
                    x_offset (int): X-coordinate of the top-left corner of the region.
                    y_offset (int): Y-coordinate of the top-left corner of the region.
                    whale_w (int): Width of the whale object.
                    whale_h (int): Height of the whale object.
                    sea_hue_range (tuple): Range of hue values for sea in HSV space (default: 90-140).
                    min_sea_ratio (float): Minimum fraction of pixels in the region that must match the sea color (default: 0.5).

                Returns:
                    bool, np.array: True if the region is sea-like, and the average HSV color of the region.
                """
                # Extract the region of interest
                region = background_image[y_offset:y_offset + whale_h, x_offset:x_offset + whale_w]

                # Convert the region to HSV color space
                hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

                # Define the sea-like color range
                lower_sea = np.array([sea_hue_range[0], 50, 50])  # Lower bound: hue, saturation, value
                upper_sea = np.array([sea_hue_range[1], 255, 255])  # Upper bound

                # Create a mask for pixels within the sea-like range
                sea_mask = cv2.inRange(hsv_region, lower_sea, upper_sea)

                # Calculate the fraction of sea-like pixels
                sea_ratio = np.sum(sea_mask > 0) / (whale_w * whale_h)

                # Calculate the average color in HSV space
                avg_hsv_color = np.mean(hsv_region, axis=(0, 1))

                # Check if the sea-like pixel ratio is sufficient
                return sea_ratio >= min_sea_ratio

            region = background_image[y_offset:y_offset+whale_h,x_offset:x_offset+whale_w]
            avg_color = np.mean(region,axis=(0,1))
            b,g,r = avg_color
            if (not specialCare):
                if(b>r and g>r ):
                    return True,avg_color # No overlap and match color
            else:
                # Checking for special care background photo horizon position not in the middle
                if (x_offset < bg_w*0.2 or x_offset > bg_w*0.8 ):
                    if (is_sea_region(background_image,x_offset,y_offset,whale_w,whale_h)):
                        return True, avg_color
            return False,None  #No match color


        # Try attempts random placements until no overlap is found
        for _ in range(attempts):
            x_offset = random.randint(0, bg_w - whale_w)
            y_offset = random.randint(vertical_min, vertical_max)
            flag , avg_color = check_no_overlap_and_color(x_offset, y_offset,cur_att)
            if flag:
                break
        else:
            print(f"Could not find a suitable position after {attempts} attempts, output photo will not save.")
            return False


        def process_whale_obj(whale_image_temp,background_image_temp):
            """
              Processes the whale image to integrate it into the background image while ensuring visual realism and applying augmentations.
              The function incorporates adjustments for size, color, texture, rotation, and ripple effects.

              Parameters:
              - whale_image_temp: The whale image in RGBA format to be processed.
              - background_image_temp: The background image where the whale will be placed.

              Workflow:
              1. Adjust the whale's size based on its position and camera perspective
              2. Match the whale's brightness and color to the background for realism.
              3. Apply optional augmentations and blending
              4. Add sea texture blending and wave-like effects to the whale's bottom region.
              5. Add ripple effects to the sea region below the whale to simulate water disturbance.
              6. Return the processed whale image and updated background image.

              Returns:
              - The processed whale image with adjustments applied.
              - The background image with ripple effects integrated.
              """
            # Extract dimensions of the background image
            bg_h_a, bg_w_a = background_image_temp.shape[:2]

            # Adjusting whale size according to the position set for it to be pasted in
            def adjust_size_over_horizon(whale_image_temp_size):
                """
                Adjusts the size of the whale image based on its position relative to the horizon and its proximity
                to the edges of the background. The size adjustment considers the y-position (horizon),
                x-position , and camera location.

                Parameters:
                - whale_image_temp_size: The original whale image to be resized.

                Returns:
                - Resized whale image based on calculated size adjustments.
                """
                # Adjust based on the y position (horizon)
                relative_position = (y_offset - horizon_y) / (bg_h_a - horizon_y)
                ratio = max(0.6, relative_position)
                fig_whale_size_adj = (int(ratio * fig_whale_size[0]), int(ratio * fig_whale_size[1]))

                # Adjust based on the x position (edges) and camara location
                camara_location = background_path[-5]
                if camara_location == 'c':  # Camera in the center
                    edge_threshold = 0.1  # 10% of the width on both sides
                    if x_offset < edge_threshold * bg_w_a or x_offset + fig_whale_size_adj[0] > (
                            1 - edge_threshold) * bg_w_a:
                        # Reduce the size by 5% if near the edges
                        fig_whale_size_adj = (int(fig_whale_size_adj[0] * 0.95), int(fig_whale_size_adj[1] * 0.95))

                elif camara_location == 'r':  # Camera on the right
                    # Slight size increase based on x position
                    relative_x = 1 - (x_offset / bg_w_a)  # Relative position, x=0 is largest, x=bg_w_a is smallest
                    size_factor = 1 + (0.2 * relative_x)  # Up to 20% increase
                    fig_whale_size_adj = (int(fig_whale_size_adj[0] * size_factor), int(fig_whale_size_adj[1] * size_factor))

                elif camara_location == 'l':  # Camera on the left
                    # Slight size increase based on x position
                    relative_x = x_offset / bg_w_a  # Relative position, x=0 is smallest, x=bg_w_a is largest
                    size_factor = 1 + (0.2 * relative_x)  # Up to 20% increase
                    fig_whale_size_adj = (int(fig_whale_size_adj[0] * size_factor), int(fig_whale_size_adj[1] * size_factor))

                # Resize the whale image
                whale_image_temp_size = cv2.resize(whale_image_temp_size, fig_whale_size_adj)
                return whale_image_temp_size

            #Fitting whales color to blend in background
            def adjust_whale_colors(whale_image_temp_c, avg_color_bg,blend_ratio = 0.85):
                """
                Adjust the whale's brightness and colors to match the background region's ambient tones.

                Parameters:
                - whale_image: The whale image in RGBA format.
                - avg_color_bg: A 1D NumPy array representing the average color (BGR) of the target background region.
                - blend_ratio: The ratio for blending whale and background colors (default: 0.85).

                Returns:
                - Adjusted whale image with brightness and colors matching the background.
                """
                # Extract the alpha channel
                alpha_channel = whale_image_temp_c[:, :, 3]
                mask = alpha_channel > 0

                # Extract RGB channels of non-transparent pixels
                whale_colors = whale_image_temp_c[mask][:, :3]  # Non-transparent pixels (B, G, R)
                # Exclude extreme pixels using interquartile range (IQR)
                q1 = np.percentile(whale_colors, 25, axis=0)  # 25th percentile
                q3 = np.percentile(whale_colors, 75, axis=0)  # 75th percentile
                iqr_mask = (whale_colors >= q1) & (whale_colors <= q3)
                valid_whale_colors = whale_colors[np.all(iqr_mask, axis=1)]  # Exclude outliers

                # Calculate brightness of whale and background
                whale_brightness = np.mean(valid_whale_colors)
                background_brightness = np.mean(avg_color_bg)

                # Calculate the dynamic adjustment factor
                adjustment_factor = background_brightness / whale_brightness
                adjustment_factor = np.clip(adjustment_factor, 0.5, 1.2)  # Clamp to avoid extreme scaling

                # Step 1: Adjust brightness dynamically
                for c in range(3):  # Iterate over B, G, R channels
                    whale_image_temp_c[:, :, c][mask] = np.clip(
                        whale_image_temp_c[:, :, c][mask].astype(np.float32) * adjustment_factor, 0, 255
                    ).astype(np.uint8)

                # Step 2: Match ambient color tones
                for c in range(3):  # Iterate over B, G, R channels
                    whale_image_temp_c[:, :, c][mask] = np.clip(
                        whale_image_temp_c[:, :, c][mask].astype(np.float32) * blend_ratio + avg_color_bg[c] * (1-blend_ratio), 0, 255
                    ).astype(np.uint8)

                return whale_image_temp_c

            # Applying Augmentation
            def apply_color_augmentation(whale_image_temp_aug):
                """
                  Applies random color augmentations to the whale image for data augmentation purposes.
                  This includes adjustments to brightness and contrast.
                  applied with a (30%) probability.

                  Parameters:
                  - whale_image_temp_aug: The whale image in RGBA format.

                  Returns:
                  - Whale image with augmented brightness and contrast.
                  """
                # brightness adjustment
                if np.random.rand() < 0.3:
                    brightness_factor = random.uniform(0.95, 1.05)
                    whale_image_temp_aug[:, :, :3] = np.clip(whale_image_temp_aug[:, :, :3] * brightness_factor, 0, 255)

                # contrast adjustment
                if np.random.rand() < 0.3:
                    contrast_factor = random.uniform(0.9, 1.1)
                    whale_image_temp_aug[:, :, :3] = np.clip(whale_image_temp_aug[:, :, :3] * contrast_factor, 0, 255)

                return whale_image_temp_aug

            def apply_noise(whale_image_temp_bn):
                """
                    Adds Gaussian noise to the whale image to enhance realism and variability in the dataset.
                    Noise is applied only to non-transparent regions of the whale image.
                    applied with a (20%) probability.

                    Parameters:
                    - whale_image_temp_bn: The whale image in RGBA format.

                    Returns:
                    - Whale image with added Gaussian noise.
                    """
                mean = 0
                sigma = 25

                if random.random() < 0.2:
                    # Extract dimensions
                    row, col, ch = whale_image_temp_bn.shape

                    # Generate Gaussian noise
                    gauss = np.random.normal(mean, sigma, (row, col, 3))  # Noise for RGB channels only

                    # Extract alpha channel as a mask (non-zero alpha means part of the object)
                    alpha_mask = whale_image_temp_bn[:, :, 3] > 0

                    # Apply noise only to the object
                    whale_image_temp_bn[:, :, :3][alpha_mask] = np.uint8(
                        np.clip(whale_image_temp_bn[:, :, :3][alpha_mask] + gauss[alpha_mask], 0, 255)
                    )

                return whale_image_temp_bn

            def apply_rotating_flipping(whale_image_temp_fr,x_offset,y_offset):
                """
                    Rotates and flips the whale image for data augmentation.
                    Rotation includes creating a larger canvas to avoid cropping the image, and flipping is applied horizontally only.
                    Rotation applied with a (20%) probability, flip applied with a (50%) probability.

                    Parameters:
                    - whale_image_temp_fr: The whale image in RGBA format.
                    - x_offset: Horizontal position of the whale image on the background.
                    - y_offset: Vertical position of the whale image on the background.

                    Returns:
                    - Rotated and/or flipped whale image.
                    - Angle of rotation (None if no rotation applied).
                  """
                angle = None
                whale_h_r, whale_w_r = whale_image_temp_fr.shape[:2]
                # Rotating and flipping
                if np.random.rand() < 0.2:
                    angle = random.uniform(-10, 10)
                    #getting the new sizes of the whale object picture
                    bounding_box = (x_offset, y_offset, x_offset + whale_w_r, y_offset + whale_h_r)
                    new_sizes = update_bbox(bounding_box,angle)
                    new_width = max(whale_w_r,new_sizes[2]-new_sizes[0])
                    new_height = max(whale_h_r,new_sizes[3]-new_sizes[1])

                    #check if whale figure is out of bonds
                    if (y_offset + new_height > bg_h_a or x_offset + new_width > bg_w_a):
                        return whale_image_temp_fr,None

                    # Create a larger canvas with transparency (alpha=0)
                    expanded_image = np.zeros((new_height, new_width, 4), dtype=whale_image_temp_fr.dtype)
                    center_x, center_y = new_width // 2, new_height // 2
                    start_x = center_x - whale_w_r // 2
                    start_y = center_y - whale_h_r // 2

                    # Place the original image in the center of the larger canvas
                    expanded_image[start_y:start_y + whale_h_r, start_x:start_x + whale_w_r] = whale_image_temp_fr

                    # Rotate the image
                    center = (new_width // 2, new_height // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    whale_image_temp_fr = cv2.warpAffine(
                        expanded_image,
                        M,
                        (new_width, new_height))

                if np.random.rand() < 0.5:
                    whale_image_temp_fr = cv2.flip(whale_image_temp_fr, 1)

                return whale_image_temp_fr,angle

            def apply_blur(whale_image_temp_bb):
                """
                    Applies a Gaussian blur to the RGB channels of the whale image for realism,
                    while preserving the alpha channel (transparency).

                    Parameters:
                    - whale_image_temp_bb: The whale image in RGBA format.

                    Returns:
                    - Whale image with blurred RGB channels.
                    """
                blur_strength = (3,3)
                # Extract the alpha channel (transparency)
                alpha_channel = whale_image_temp_bb[:, :, 3]

                # Apply Gaussian blur to the RGB channels
                blurred_rgb = cv2.GaussianBlur(whale_image_temp_bb[:, :, :3], blur_strength, 0)

                # Combine the blurred RGB channels with the original alpha channel
                blurred_image = np.dstack((blurred_rgb, alpha_channel))

                return blurred_image

            def apply_sea_texture(whale_image_temp_b,sea_texture,blend_area = 0.3):
                """
                Adjust the bottom region of the whale object to make it more blue,
                blending with the average color of the target placement area.

                Parameters:
                - whale_image: The whale image with an alpha channel (RGBA format).
                -sea_texture: A region where the whale is going to be pasted for sea texture
                -blend_area : The fraction of the whale's vertical height to blend (default: 0.3).

                Returns:
                - Modified whale image with adjusted bottom region.
                - The top row of the blending region.
                """
                # Strength of the blend
                base_blend_ratio = 0.4  # Minimum blend ratio
                max_blend_ratio = 0.9  # Maximum blend ratio

                # Extract the alpha channel
                alpha_channel = whale_image_temp_b[:, :, 3]

                # Find the bottom-most row where alpha > 40
                non_transparent_rows = np.where(alpha_channel > 40)[0]

                try:
                    bottom_row = non_transparent_rows[-1]
                except:
                    print("Warning : didn't apply color texture")
                    return whale_image_temp_b,None

                top_row = int(bottom_row - blend_area * (bottom_row - non_transparent_rows[0]))
                # Get the width of the image
                _, img_width = alpha_channel.shape

                # Define wave parameters
                wave_amplitude = 0.2 * (bottom_row - top_row)  # Amplitude of the wave (vertical height)
                wave_frequency = random.randint(2,4)  # Number of wave cycles across the width
                random_phase = np.random.uniform(0, 2 * np.pi)  # Random phase shift for the sine wave
                x = np.linspace(0, 2 * np.pi, img_width) +random_phase  # Add phase shift to the wave

                # Calculate wave offsets (vertical displacement at each column)
                wave_offsets = (np.sin(x * wave_frequency) * wave_amplitude).astype(int)

                # Blend the area with wave-like pattern
                for y in range(top_row, bottom_row +1):
                    for x_pos in range(img_width):
                        # Calculate the adjusted row based on the wave pattern
                        adjusted_y = y + wave_offsets[x_pos]

                        # Ensure adjusted_y stays within valid bounds
                        if adjusted_y < 0 or adjusted_y >= alpha_channel.shape[0]:
                            continue

                        # Linearly adjust blend ratio from base to max
                        blend_ratio = base_blend_ratio + (max_blend_ratio - base_blend_ratio) * (
                                    (adjusted_y - top_row) / (bottom_row - top_row))
                        try:
                            # Blend the whale object with the sea texture at the adjusted row
                            for c in range(3):  # Iterate over B, G, R channels
                                whale_image_temp_b[adjusted_y, x_pos, c] = np.clip(
                                    whale_image_temp_b[adjusted_y, x_pos, c] * (1 - blend_ratio)
                                    + sea_texture[adjusted_y, x_pos, c] * blend_ratio,
                                    0,
                                    255,
                                )
                        except:
                            return whale_image_temp_b,None
                return whale_image_temp_b, top_row

            #Get parameters for ripple effect
            def get_shape_param(whale_image_temp_s):
                """
                Extracts shape parameters from the whale image, including its bottom row, width, and height.
                Useful parameters for applying ripple pattern around the whale object.

                Parameters:
                - whale_image_temp_s: The whale image with an alpha channel (RGBA format).

                Returns:
                - bottom_row: The bottom-most row of non-transparent pixels.
                - top_bottom_width: The maximum width of the object near its base.
                - object_height: The height of the object (non-transparent region).
                - first_non_transparent_column: The leftmost column of the non-transparent region.

                """
                # Extract the alpha channel
                alpha_channel = whale_image_temp_s[:, :, 3]

                # Find the bottom-most row where alpha > 40
                non_transparent_rows = np.where(alpha_channel > 40)[0]


                # Find the bottom-most row
                bottom_row = non_transparent_rows[-1]

                top_end_y = int(bottom_row * 0.7)

                # Analyze the alpha region
                alpha_region = alpha_channel[top_end_y:bottom_row + 1, :]
                mask_alpha = alpha_region > 20

                # Calculate width of the non-transparent region at each row
                row_nontrans_width = np.sum(mask_alpha, axis=1)
                top_bottom_width = np.max(row_nontrans_width)

                # Find the first column where alpha > 40
                first_non_transparent_column = np.where(mask_alpha)[1].min()

                # Calculate the height of the object (number of non-transparent rows)
                object_height = bottom_row - non_transparent_rows[0]

                # Ensure the height does not exceed the width
                if(object_height>top_bottom_width): object_height=top_bottom_width

                return bottom_row,top_bottom_width, object_height,first_non_transparent_column

            def get_sea_region_shape(y_offset,x_offset,oval_height_offset,whale_h_a,whale_w_a,first_non_transparent_column,bottom_width):
                """
                   Calculates the bounding box for the sea region under the whale image. This region is used for applying
                   ripple effect.

                   Parameters:
                   - y_offset: The vertical offset of the whale on the background.
                   - x_offset: The horizontal offset of the whale on the background.
                   - oval_height_offset: Offset height for the oval region under the whale.
                   - whale_h_a: Height of the whale image.
                   - whale_w_a: Width of the whale image.
                   - first_non_transparent_column: The first non-transparent column in the whale image.
                   - bottom_width: The width of the whale's bottom region.

                   Returns:
                   - A tuple defining the sea region's bounding box (start and end points for y and x).
                   - x_center: The center of the region in the x-direction.
                   """
                # Calculate vertical bounds of the sea region
                y_point_start = (y_offset + oval_height_offset)
                y_point_end = y_point_start+whale_h_a//2

                # Calculate the horizontal center of the whale's bottom
                x_center = x_offset+ (first_non_transparent_column+bottom_width//2)

                # Calculate horizontal bounds of the sea region
                x_point_start = max(0,x_center-whale_w_a)
                x_point_end = x_center+int(whale_w_a)


                return (y_point_start,y_point_end,x_point_start,x_point_end),x_center

            #Applying ripple effect
            def apply_ripple_effect(sea_region,avg_color,bottom_width, ripple_height):
                """
                    Applies a ripple effect to a specified sea region to enhance visual realism.
                    The effect is created using sinusoidal waves within an elliptical region.

                    Parameters:
                    - sea_region: The region of the sea where the ripple effect is applied.
                    - avg_color: The average color of the sea region (BGR format).
                    - bottom_width: The width of the whale's bottom region, used to define the ripple width.
                    - ripple_height: The vertical height of the ripple effect.

                    Returns:
                    - Blended region with the ripple effect applied.
                """

                # Number of ripples in the effect
                ripple_count=5
                # Intensity of the ripple effect
                intensity = 50
                # Ratio for blending the ripples with the original sea texture

                blend_ratio = 0.5
                # Get region dimensions
                region_height, region_width = sea_region.shape[:2]

                # Define the center of the ellipse (center-top of the ripple region)
                center_x = region_width // 2
                center_y = region_height//5  # Start from the top of the region

                # Define the semi-major and semi-minor axes of the ellipse
                a = bottom_width   # Semi-major axis (half of the bottom width)
                b = ripple_height  # Semi-minor axis (half of the ripple height)
                # Define the semi-major and semi-minor axes of the inner excluded ellipse
                inner_a = bottom_width / 2   # Inner semi-major axis (30% of the outer major axis)
                inner_b =  ripple_height / 2 # Inner semi-minor axis (30% of the outer minor axis)

                # Create a meshgrid for the region
                y, x = np.meshgrid(np.arange(region_height), np.arange(region_width), indexing="ij")

                # Calculate the outer ellipse mask
                outer_ellipse_mask = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1

                # Calculate the inner ellipse mask
                inner_ellipse_mask = ((x - center_x) ** 2 / inner_a ** 2 + (y - center_y) ** 2 / inner_b ** 2) <= 1

                # Combine the masks to exclude the center
                ellipse_mask = outer_ellipse_mask & ~inner_ellipse_mask

                if not np.any(ellipse_mask):
                    print("Warning: The ellipse mask is empty. No ripples will be applied.")
                    return sea_region

                # Normalize the distance within the ellipse region
                distance = np.sqrt((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2)
                normalized_distance = np.clip(distance, 0, 1)

                # Create the ripple pattern
                ripples = np.sin(normalized_distance * ripple_count * 2 * np.pi) * (1 - normalized_distance)

                # Apply the ripple mask
                ripples *= ellipse_mask

                # Scale the ripples to the desired intensity
                ripple_pattern = (ripples * intensity).astype(np.float32)

                # Convert the ripple pattern into a 3-channel texture
                ripple_texture = cv2.merge([ripple_pattern] * 3)

                # Incorporate the average color into the ripples
                avg_color_texture = np.full_like(ripple_texture, avg_color, dtype=np.float32)
                blended_ripple_texture = cv2.addWeighted(avg_color_texture, 1, ripple_texture, blend_ratio, 0)

                # Blend the ripple texture with the original region
                blended_region = (
                        sea_region.astype(np.float32) * (1 - blend_ratio) + blended_ripple_texture * blend_ratio
                )

                # Clip and return the result
                blended_region = np.clip(blended_region, 0, 255).astype(np.uint8)
                return blended_region

            def transform_ripple_effect(background_image_temp_ripple, ripple_region, angle_2_ripple, ripple_region_loc):
                """
                   Rotates and integrates the ripple effect into the background image. Ensures alignment with
                   the angle of the whale object.

                   Parameters:
                   - background_image_temp_ripple: The background image where ripples will be integrated.
                   - ripple_region: The ripple effect region to be applied.
                   - angle_2_ripple: The angle by which the ripple region is rotated.
                   - ripple_region_loc: The bounding box (top-left and bottom-right) of the ripple region in the background.

                   Returns:
                   - Updated background image with the ripple effect applied.
                """
                reg_h, reg_w = ripple_region.shape[:2]

                # Add alpha channel to ripple region if it doesn't exist
                if ripple_region.shape[2] != 4:
                    alpha_channel_temp = np.full((reg_h, reg_w), 255, dtype=np.uint8)
                    ripple_region = np.dstack((ripple_region, alpha_channel_temp))

                #Add 5-pixel gradient to edges of the ripple region
                for i in range(5):
                    alpha_value = int(255 * (i / 5))
                    ripple_region[:, i, 3] = np.minimum(ripple_region[:, i, 3], alpha_value)  # Left edge
                    ripple_region[:, reg_w - 1 - i, 3] = np.minimum(ripple_region[:, reg_w - 1 - i, 3],
                                                                    alpha_value)  # Right edge

                # Update the bounding box for the ripple region
                ripple_region_bbox=(ripple_region_loc[2],ripple_region_loc[0],ripple_region_loc[3],ripple_region_loc[1])
                ripple_region_bbox_updated = update_bbox(ripple_region_bbox, angle_2_ripple)
                new_width = max(reg_w, ripple_region_bbox_updated[2] - ripple_region_bbox_updated[0])
                new_height = max(reg_h, ripple_region_bbox_updated[3] - ripple_region_bbox_updated[1])
                # Create a larger canvas and save the new locations
                x1_new = ripple_region_bbox_updated[0]
                y1_new = ripple_region_bbox_updated[1]
                # Check if the updated dimensions exceed the background boundaries
                # if the new demanssion is out of the background photo shape return the original without implment ripple
                if(y1_new+new_height >bg_h_a or x1_new+new_width>bg_w_a):
                    return background_image_temp_ripple

                # Update the new bounding box coordinates
                y2_new = ripple_region_bbox_updated[1]+new_height
                x2_new = ripple_region_bbox_updated[0]+new_width

                # Extract the new region from background and add an alpha channel
                expanded_image = background_image_temp_ripple[y1_new:y2_new,x1_new:x2_new]
                alpha_channel_temp = np.full((new_height, new_width), 0, dtype=np.uint8)
                expanded_image = np.dstack((expanded_image, alpha_channel_temp))

                # Center the ripple region within the expanded canvas
                center_x, center_y = new_width // 2, new_height // 2
                start_x = center_x - reg_w // 2
                start_y = center_y - reg_h // 2

                # Place the original image in the center of the larger canvas
                expanded_image[start_y:start_y + reg_h, start_x:start_x + reg_w] = ripple_region

                # Rotate the image
                center = (new_width // 2, new_height // 2)
                M_2 = cv2.getRotationMatrix2D(center, angle_2_ripple, 1.0)
                ripple_region_rotated = cv2.warpAffine(
                    expanded_image,
                    M_2,
                    (new_width, new_height))

                # Overlay ripple region on background
                alpha_channel = ripple_region_rotated[:, :, 3]
                alpha_s = alpha_channel / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    background_image_temp_ripple[y1_new:y2_new, x1_new:x2_new, c] = (
                            alpha_s * ripple_region_rotated[:, :, c] +
                            alpha_l * background_image_temp_ripple[y1_new:y2_new, x1_new:x2_new, c]
                    )

                return background_image_temp_ripple

            # Step 1: Adjust the whale's size based on its position and camera perspective
            whale_image_temp = adjust_size_over_horizon(whale_image_temp)
            # Update dimensions after resizing
            whale_h_a, whale_w_a = whale_image_temp.shape[:2]

            # Step 2: Match whale colors and brightness to the background
            whale_image_temp = adjust_whale_colors(whale_image_temp,avg_color)

            # Step 3: Apply optional augmentations and blending
            if not (photoshop_whale):
                # Apply random color augmentation
                whale_image_temp = apply_color_augmentation(whale_image_temp)

                # Apply random Gaussian noise for augmentation
                whale_image_temp = apply_noise(whale_image_temp)

                # Blend the whale's bottom with the sea texture
                # Extract the region in the background where the whale will be placed
                region = background_image_temp[y_offset:y_offset+whale_h_a,x_offset:x_offset+whale_w_a]
                if with_cover :
                    whale_image_temp,oval_height_offset= apply_sea_texture(whale_image_temp,region)
                else:
                    whale_image_temp, oval_height_offset = apply_sea_texture(whale_image_temp, region,blend_area=0.1)

                # Apply blur to smooth the whale image
                whale_image_temp = apply_blur(whale_image_temp)

                # Randomly rotate and flip the whale image
                whale_image_temp, angle = apply_rotating_flipping(whale_image_temp,x_offset,y_offset)

                # Update whale dimensions after rotation
                whale_h_a, whale_w_a = whale_image_temp.shape[:2]

                # Step 4: Apply ripple effects to the sea region below the whale
                if (with_ripple and (not specialCare)):
                    # Extract shape parameters of the whale object
                    first_non_transparent_row, bottom_width, oval_height,first_non_transparent_column = get_shape_param(whale_image_temp)
                    # Get sea region location from shap parameters
                    sea_region_loc,x_center = get_sea_region_shape(y_offset,x_offset,oval_height_offset,whale_h_a,whale_w_a,first_non_transparent_column,bottom_width)
                    # Extract the sea region from the background image
                    sea_region = background_image_temp[sea_region_loc[0]:sea_region_loc[1],sea_region_loc[2]:sea_region_loc[3]]
                    # Apply ripple effect to the sea region
                    sea_region = apply_ripple_effect(sea_region,avg_color,bottom_width,oval_height)
                    # Integrate the ripple effect into the background
                    if(angle is not None):
                        background_image_temp = transform_ripple_effect(background_image_temp,sea_region,angle,sea_region_loc)
                    else:
                        background_image_temp[sea_region_loc[0]:sea_region_loc[1], sea_region_loc[2]:sea_region_loc[3]] =sea_region


            else:

                # When photoshop_whale is enabled, skip some augmentations
                region = background_image_temp[y_offset:y_offset + whale_h_a, x_offset:x_offset + whale_w_a]
                whale_image_temp, oval_height_offset = apply_sea_texture(whale_image_temp, region)
                whale_image_temp, angle = apply_rotating_flipping(whale_image_temp, x_offset, y_offset)

            # Return the processed whale image and the updated background
            return whale_image_temp, background_image_temp

        def update_bbox(bbox, angle):
            """
                Updates a bounding box after rotating it around it's center by a specified angle.
                The function calculates the new coordinates of the bounding box that encloses the
                rotated region.

                Parameters:
                - bbox: A tuple representing the bounding box in the format (x1, y1, x2, y2), where:
                    - (x1, y1): Top-left corner.
                    - (x2, y2): Bottom-right corner.
                - angle: The angle (in degrees) to rotate the bounding box around its center.

                Returns:
                - A new bounding box tuple (x1_new, y1_new, x2_new, y2_new) that tightly encloses
                  the rotated region.
                """

            x1, y1, x2, y2 = bbox
            # Rotate around the center of the bounding box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # Generate a 2D rotation matrix to rotate around the center
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Define the four corners of the bounding box as a NumPy array
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
            # Apply the rotation transformation to the corners
            rotated_corners = cv2.transform(np.array([corners]), rot_matrix)[0]

            # Compute the new bounding box that encloses the rotated corners
            x1_new = int(np.min(rotated_corners[:, 0]))
            y1_new = int(np.min(rotated_corners[:, 1]))
            x2_new = int(np.max(rotated_corners[:, 0]))
            y2_new = int(np.max(rotated_corners[:, 1]))

            bbox = (x1_new, y1_new, x2_new, y2_new)
            return bbox



        # Process whale object -Augmentation and Realism
        whale_image,background_image = process_whale_obj(whale_image,background_image)
        whale_h_updated, whale_w_updated = whale_image.shape[:2]

        # Bounding box
        bounding_box = (x_offset, y_offset, x_offset + whale_w_updated, y_offset + whale_h_updated)

        # Overlay whale on background
        alpha_channel = whale_image[:, :, 3]
        alpha_s = alpha_channel / 255.0
        alpha_l = 1.0 - alpha_s
        try:
            for c in range(0, 3):
                background_image[y_offset:y_offset + whale_h_updated, x_offset:x_offset + whale_w_updated, c] = (
                        alpha_s * whale_image[:, :, c] +
                        alpha_l * background_image[y_offset:y_offset + whale_h_updated, x_offset:x_offset + whale_w_updated, c]
                )
        except:
            print(f"Process fail to create photo, output photo will not save.")
            return False

        # Create Annotation file
        annotation_2_yolo(bounding_box,bg_w,bg_h,annotation_path)

        # Save the image without bbox and horizon line
        if(output_path!=None):
            cv2.imwrite(output_path, background_image)

        # Draw the bounding box on the background
        cv2.rectangle(background_image,
                      (bounding_box[0], bounding_box[1]),  # Top-left corner
                      (bounding_box[2], bounding_box[3]),  # Bottom-right corner
                      (0, 255, 0), 1)  # Green color and thickness of 2
        cv2.line(background_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Save the image with bbox
        if (with_bbox):
            cv2.imwrite(output_path_bbox, background_image)

        # Returning positive for saving output photo
        return True
    else:
        print("Horizon line not detected, output photo will not save.")
        return False

def annotation_2_yolo(bounding_box,img_w,img_h,annotation_path,clean_annot=False):
    """
      Converts a bounding box to YOLO annotation format and writes it to a specified file.
     The YOLO format normalizes bounding box coordinates relative to the image dimensions.
      Each line in a YOLO annotation file contains:
          class_id x_center y_center width height

      Parameters:
      - bounding_box: A tuple representing the bounding box in (x_min, y_min, x_max, y_max) format.
      - img_w: Width of the image.
      - img_h: Height of the image.
      - annotation_path: The file path where the YOLO annotation will be saved.
      - clean_annot: A boolean flag. If `True`, clears the annotation file without writing any data.

      Writes:
      - A line to the `annotation_path` file in YOLO format.
      """
    if clean_annot:
        with open(annotation_path, "w") as f:
            return
    bbox_x_min = bounding_box[0]
    bbox_y_min = bounding_box[1]
    bbox_x_max = bounding_box[2]
    bbox_y_max = bounding_box[3]

    x_center = ((bbox_x_min+bbox_x_max)/2)/img_w
    y_center = ((bbox_y_min + bbox_y_max) / 2) / img_h
    n_width = (bbox_x_max - bbox_x_min) / img_w
    n_height = (bbox_y_max - bbox_y_min) / img_h


    with open(annotation_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {n_width:.6f} {n_height:.6f}")

def save_background_lines(background_img,lines,background_line_path):
    if lines is not None:
        # Draw all detected lines on a copy of the background image
        lines_image = background_img.copy()
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines

        # Save the image with lines
        cv2.imwrite(background_line_path, lines_image)
        print(f"Saved image with lines: {background_line_path}")
    else:
        print("No lines detected.")

def clean_background_photo_maker(background_img_path,final_image_path,annotation_path):
    """
        Creates a clean background photo and its corresponding annotation file.

        Parameters:
        - background_img_path: Path to the input background image.
        - final_image_path: Path where the background image will be saved.
        - annotation_path: Path where the cleared YOLO annotation file will be created.

    """
    # Load the background image from the specified path
    background_image = cv2.imread(background_img_path)
    # Save the loaded image to the final image path
    cv2.imwrite(final_image_path,background_image)
    # Clear the YOLO annotation file by passing `clean_annot=True` to the function
    annotation_2_yolo(0,0,0,annotation_path,True)




background_img_path = r"C:\Users\elad6\Desktop\project photos\background photos\background.jpg"
whale_in_path = r"C:\Users\elad6\Desktop\project photos\whales photos\photos2process\Whale_og_.jpg"
whale_out_path = r"C:\Users\elad6\Desktop\project photos\whales photos\whales_after_removal\0701 check without cover\whale_output_chack_cover_.png"
final_image_path_bbox = rf"C:\Users\elad6\Desktop\project photos\output photos with lines\output photos blow check\output_photo_bbox_chk_06_01.jpg"
final_image_path = rf"C:\Users\elad6\Desktop\project photos\output photos with lines\output photos blow check\output_photo_chk_14_01.jpg"
annotation_obj_path = rf"C:\Users\elad6\Desktop\project photos\photos 2 detection\labels\output_photo_delete.txt"
atmp = 20
fig_size_all = (int(2.75*35), int(2.0625*23))
fig_size_ps = (int(2.25*35), int(1.6875*23))

'''Cheking specific whale removal '''
# for j in range(182):
#     whale_in_path_temp = whale_in_path[:-4] + f'{j + 1}' + whale_in_path[-4:]
#     whale_out_path_temp = whale_out_path[:-4] + f'{j + 1}' + whale_out_path[-4:]
#     start_t = time.time()
#     whale_back_removal(in_path=whale_in_path_temp, out_path=whale_out_path_temp)
#     end_t = time.time()
#     print(f"For 1 removing background - time pass : {end_t - start_t}")

# start_t = time.time()
# whale_back_removal(in_path=whale_in_path, out_path=whale_out_path)
# end_t = time.time()
# print(f"For 1 removing background - time pass : {end_t-start_t}")

folder_path_top15= r"C:\Users\elad6\Desktop\project photos\whales photos\whales_after_removal\Chosen Whale obj after removal\all"
folder_path_top15_ps=r"C:\Users\elad6\Desktop\project photos\whales photos\whales_after_removal\Chosen Whale obj after removal\all\PhotoShop obj"
folder_path_top15_cover=r"C:\Users\elad6\Desktop\project photos\whales photos\whales_after_removal\Chosen Whale obj after removal\all\without cover impl"
folder_path_top15_ripple = r"C:\Users\elad6\Desktop\project photos\whales photos\whales_after_removal\Chosen Whale obj after removal\batch4\without ripple impl"

#creating pasted photos from folder path
path_array_whale = [folder_path_top15,folder_path_top15_ps,folder_path_top15_cover]

def ongoing_dataset_maker(path_array_whale, folder_path_background,folder_path_bckgnd_ps, fig_size_ps, fig_size_all, offset=0,sc=False):
    cnt = 0+offset
    fig_size = fig_size_all
    num_photos = 7
    ps = False
    cv = True
    rp= True
    for path_w in path_array_whale:
        if(folder_path_top15_ps==path_w):
            ps = True
            fig_size = fig_size_ps
            #folder_path_background = folder_path_bckgnd_ps
            num_photos = 8
        elif(path_w==folder_path_top15_ripple):
            rp=False
        elif(folder_path_top15_cover == path_w):
            cv=False
        for file_name in os.listdir(path_w):
            # Construct the full path to the file
            if file_name.lower().endswith(('.png')):
                whale_out_path_temp = os.path.join(path_w, file_name)
                print(f"Whale obj :\n{whale_out_path_temp}")
                print(f"from cnt: {cnt+1}")
                for file_name in os.listdir(folder_path_background):
                    # Construct the full path to the file
                    if file_name.lower().endswith(('.jpg')):
                        background_img_path_temp = os.path.join(folder_path_background, file_name)
                    for j in range (num_photos):
                        final_image_path_bbox_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch4\images\with bbox\output_photo_bbox_{cnt + 1}.jpg"
                        final_image_path_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch4\images\without bbox\output_photo_{cnt + 1}.jpg"
                        annotation_obj_path_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch4\annotation\output_photo_{cnt + 1}.txt"
                        photo_made_flag= paste_whale(background_path=background_img_path_temp, whale_path=whale_out_path_temp, output_path=final_image_path_temp, output_path_bbox = final_image_path_bbox_temp
                                    , annotation_path=annotation_obj_path_temp, fig_whale_size=fig_size, attempts=atmp,
                                    with_cover=cv, photoshop_whale=ps,with_ripple=rp, with_bbox=False,specialCare=sc)
                        if (photo_made_flag):
                            cnt+=1
                print(f"to cnt: {cnt}")
        ps = False
        cv = True
        rp = True

def last_dataset_maker(path_photos_whale, folder_path_background, fig_size_ps, fig_size_all, offset=0,sc=False):
    cnt = 0+offset
    fig_size = fig_size_all
    num_photos = 2
    path_array_whale= os.listdir(path_photos_whale)
    len_whale_path_arr = len(path_array_whale)
    cnt_whale_photo =0
    ps = False
    cv = False
    rp= True

    for file_name in os.listdir(folder_path_background):
        # Construct the full path to the file
        if file_name.lower().endswith(('.jpg')):
            background_img_path_temp = os.path.join(folder_path_background, file_name)
        if(cnt_whale_photo==len_whale_path_arr):
            cnt_whale_photo=0
        file_name_whale = path_array_whale[cnt_whale_photo]
        while not file_name_whale.lower().endswith(('.png')):
            cnt_whale_photo+=1
            if (cnt_whale_photo == len_whale_path_arr):
                cnt_whale_photo = 0
            file_name_whale = path_array_whale[cnt_whale_photo]
        whale_out_path_temp = os.path.join(path_photos_whale, file_name_whale)
        photo_made_flag= False
        cnt_while_loops = 0
        while (not photo_made_flag) :
            cnt_while_loops +=1
            if (cnt_while_loops == 2):
                break
            final_image_path_bbox_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch5\images\with bbox\output_photo_bbox_{cnt + 1}.jpg"
            final_image_path_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch5\images\without bbox\output_photo_{cnt + 1}.jpg"
            annotation_obj_path_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch5\annotation\output_photo_{cnt + 1}.txt"
            photo_made_flag= paste_whale(background_path=background_img_path_temp, whale_path=whale_out_path_temp, output_path=final_image_path_temp, output_path_bbox = final_image_path_bbox_temp
                        , annotation_path=annotation_obj_path_temp, fig_whale_size=fig_size, attempts=atmp,
                        with_cover=cv, photoshop_whale=ps,with_ripple=rp, with_bbox=False,specialCare=sc)
            if (photo_made_flag):
                cnt+=1
                cnt_whale_photo += 1
        if (cnt % 100 == 0 ):
            print (f"created : {cnt} photos.")

folder_path_bckgnd= r"C:\Users\elad6\Desktop\project photos\background photos\batch5\whales_original_dataset"
folder_path_bckgnd_ps=r"C:\Users\elad6\Desktop\project photos\background photos\batch5\whales_original_dataset\for ps"
folder_path_bckgnd_cv=r"C:\Users\elad6\Desktop\project photos\background photos\batch5\whales_original_dataset\for cv"

#last_dataset_maker(path_array_whale[2],folder_path_bckgnd_cv,fig_size_ps,fig_size_all,8373,True)
import re
background_path = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch5\images\without bbox"
background_array = os.listdir(background_path)
annotation_path = rf"C:\Users\elad6\Desktop\project photos\full dataset\output photos batch5\annotation"
annotation_array = os.listdir(annotation_path)
count=0
for file_name in background_array:
    if (count ==2 ):
        break
    if file_name.lower().endswith(('.jpg')):
        name_without_ext = file_name.rsplit('.', 1)[0]
        match = re.search(r'(\d+)$', name_without_ext)
        if match:
            number = match.group(1)
            for file_ann in annotation_array:
                name_without_ext1 = file_ann.rsplit('.', 1)[0]
                match1 = re.search(r'(\d+)$', name_without_ext1)
                if match1:
                    number1 = match1.group(1)
                if number==number1 :
                    break
                elif number1>number:
                    print(f"number missing {number}")
                    count+=1
                    break
'''Cheking specific whale paste '''

# start_t = time.time()
# paste_whale(background_path=background_img_path, whale_path=whale_out_path, output_path=final_image_path,output_path_bbox = final_image_path_bbox
#                 ,annotation_path=annotation_obj_path ,fig_whale_size=fig_size, attempts=atmp,with_bbox=False)
# end_t = time.time()
# print(f"For 1 paste whale - time pass : {end_t-start_t}")
# cnt = 8650
# folder_path_bckgnd_arr= [r"C:\Users\elad6\Desktop\project photos\background photos\batch2",
# r"C:\Users\elad6\Desktop\project photos\background photos\batch3",r"C:\Users\elad6\Desktop\project photos\background photos\batch4"]
# for folder_path_background_temp in folder_path_bckgnd_arr:
#     for file_name in os.listdir(folder_path_background_temp):
#         if file_name.lower().endswith(('.jpg')):
#             background_img_path_temp = os.path.join(folder_path_background_temp, file_name)
#         else :
#             continue
#         for j in range(10):
#             final_image_path_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\only background\images\output_photo_{cnt + 1}.jpg"
#             annotation_obj_path_temp = rf"C:\Users\elad6\Desktop\project photos\full dataset\only background\annotation\output_photo_{cnt + 1}.txt"
#             clean_background_photo_maker(background_img_path_temp,final_image_path_temp,annotation_obj_path_temp)
#             cnt+=1
#             if cnt%100 ==0 : print(cnt)

def photo_dataset_maker(background_img_path, whale_in_path, whale_out_path, output_path,
                         num_whale_pht_per_back=5, precent_clean_back_pht=0, output_bbox_pht=False):
    """
    This function generates a dataset of photos by removing whale backgrounds,
    pasting whales on various background images, and optionally generating clean
    background photos with empty annotations.

    Parameters:
        background_img_path (str): Path to the folder containing background images.
        whale_in_path (str): Path to the folder containing whale images with original backgrounds.
        whale_out_path (str): Path to the folder where whale images without backgrounds will be saved.
        output_path (str): Path to the folder where output images and annotations will be saved.
        num_whale_pht_per_back (int): Number of whale photos to paste on each background (default is 5).
        precent_clean_back_pht (float): Percentage of clean background photos to generate (default is 0).
        output_bbox_pht (bool): Flag to indicate whether to output images with bounding boxes (default is False).
    """

    # Initialize variables for processing
    atmp = 20  # Number of attempts for pasting the whale
    fig_size = (int(2.75 * 35), int(2.0625 * 23))  # Size of the whale images
    arr_time_whale_rem = []  # List to store time taken for whale background removal
    arr_time_whale_paste = []  # List to store time taken for pasting whale
    counter_img_back = 0  # Counter for background images
    counter_img_whale = 0  # Counter for whale images
    number_of_bckgrnd_photos = 0  # Counter for total number of background photos processed
    obj_whale_name = 'whale_output_rembg.png'  # Template for whale output name

    # Process whale photos: remove backgrounds from whale images
    for file_name in os.listdir(whale_in_path):
        if file_name.lower().endswith('.jpg'):
            whale_in_path_temp = os.path.join(whale_in_path, file_name)
            obj_whale_name_temp = obj_whale_name[:-4] + f'{counter_img_whale + 1}' + obj_whale_name[-4:]
            whale_out_path_temp = os.path.join(obj_whale_name_temp, obj_whale_name)
            print(f"Processing Whale photo :\n{file_name}")
            start_t = time.time()
            whale_back_removal(in_path=whale_in_path_temp, out_path=whale_out_path_temp)
            end_t = time.time()

            # Track time taken for background removal
            if (end_t - start_t >= 1):
                arr_time_whale_rem.append(end_t - start_t)
            counter_img_whale += 1

    # Output the mean time for whale background removal
    print("Finished processing original whale photos.")
    arr_time_whale_rem_np = np.array(arr_time_whale_rem)
    if len(arr_time_whale_rem_np) != 0:
        print(f"For {counter_img_whale} photos of whale removing background, mean time passed: {np.mean(arr_time_whale_rem_np)} sec")

    # Create necessary folders for saving output images and annotations
    images_folder = os.path.join(output_path, "images")
    images_with_bbox = os.path.join(images_folder, "with bbox")
    images_without_bbox = os.path.join(images_folder, "without bbox")
    annotation_folder = os.path.join(output_path, "annotation")

    # Create the folder structure if it doesn't exist
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(images_with_bbox, exist_ok=True)
    os.makedirs(images_without_bbox, exist_ok=True)
    os.makedirs(annotation_folder, exist_ok=True)

    # Process whale pasting: paste the whale on background images
    for file_name in os.listdir(whale_out_path):
        if file_name.lower().endswith('.png'):
            whale_out_path_temp = os.path.join(whale_out_path, file_name)
            print(f"Processing generated photos for whale photo: {file_name}, numbers:")
            print(f"Output photos start from number: {counter_img_back + 1}")

            # Loop over background images and paste the whale onto them
            for file_name in os.listdir(background_img_path):
                if file_name.lower().endswith('.jpg'):
                    background_img_path_temp = os.path.join(background_img_path, file_name)
                number_of_bckgrnd_photos += 1
                for j in range(num_whale_pht_per_back):
                    final_image_path_bbox_temp = os.path.join(output_path, f"images/with bbox/output_photo_bbox_{counter_img_back + 1}.jpg")
                    final_image_path_temp = os.path.join(output_path, f"images/without bbox/output_photo_{counter_img_back + 1}.jpg")
                    annotation_obj_path_temp = os.path.join(output_path, f"annotation/output_photo_{counter_img_back + 1}.txt")

                    # Paste whale and save the final image and annotation
                    photo_made_flag=paste_whale(background_path=background_img_path_temp, whale_path=whale_out_path_temp,
                                 output_path=final_image_path_temp, output_path_bbox=final_image_path_bbox_temp,
                                 annotation_path=annotation_obj_path_temp, fig_whale_size=fig_size, attempts=atmp,
                                 with_bbox=output_bbox_pht)
                    # Cheking if photo do saved
                    if (photo_made_flag):
                        counter_img_back += 1
            print(f"Output photos end in number: {counter_img_back}")

    # Create clean background photos with empty annotations if needed
    if precent_clean_back_pht != 0:
        num_clean_back_pht = int(counter_img_back * precent_clean_back_pht)
        num_pht_ber_bck_clean = max(1, num_clean_back_pht // number_of_bckgrnd_photos)
        for file_name in os.listdir(background_img_path):
            if file_name.lower().endswith('.jpg'):
                background_img_path_temp = os.path.join(background_img_path, file_name)
            for r in range(num_pht_ber_bck_clean):
                final_image_path_temp = os.path.join(output_path, f"images/without bbox/output_photo_{counter_img_back + 1}.jpg")
                annotation_obj_path_temp = os.path.join(output_path, f"annotation/output_photo_{counter_img_back + 1}.txt")

                # Create clean background photo
                clean_background_photo_maker(background_img_path_temp, final_image_path_temp, annotation_obj_path_temp)
                counter_img_back += 1



