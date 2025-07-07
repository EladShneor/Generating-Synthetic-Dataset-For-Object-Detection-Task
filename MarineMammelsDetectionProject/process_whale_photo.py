import os.path
from rembg import remove
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import random


def whale_back_removal(in_path, out_path):
    # Check if the output file already exists
    if os.path.exists(out_path):
        print(f"Whale photo after removal already exists, skipping background removal.")
        return

    with open(in_path, "rb") as input_file:
        input_image = input_file.read()

    output_image = remove(input_image)
    img = Image.open(io.BytesIO(output_image)).convert("RGBA")

    def add_splash(base_img, position, x_offset_rng, y_offset_rng, droplet_count=100):
        # Create a blank overlay for the splash
        splash_overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(splash_overlay)
        y_offset_rng = max(20, y_offset_rng)
        # Generate random droplets
        for _ in range(droplet_count):
            radius = random.randint(5, 8)  # Random droplet size
            x_offset = random.randint(-x_offset_rng, x_offset_rng)
            y_offset = random.randint(0, y_offset_rng)
            droplet_position = (position[0] + x_offset, position[1] + y_offset)
            color = (255, 255, 255, random.randint(130, 200))  # Semi-transparent white
            draw.ellipse([
                (droplet_position[0] - radius, droplet_position[1] - radius),
                (droplet_position[0] + radius, droplet_position[1] + radius)
            ], fill=color)

        # Blur the overlay for realism
        splash_overlay = splash_overlay.filter(ImageFilter.GaussianBlur(2))

        # Composite the overlay onto the base image
        combined = Image.alpha_composite(base_img, splash_overlay)
        return combined

    # Find the bounding box
    bbox = img.getbbox()
    if bbox:
        # Crop the image to the bounding box
        cropped_img = img.crop(bbox)

        # find position part
        def find_pos_2_splash(base_img):
            _, _, _, alpha = base_img.split()
            alpha_array = np.array(alpha)

            # Find bottom-most row where alpha > 40 (non-transparent)
            non_zero_rows = np.where(alpha_array > 40)[0]
            bottom_y = int(non_zero_rows[-1])

            # Define the vertical range for analysis (top 85% of the bottom region)
            top_end_y = int(bottom_y * 0.85)

            alpha_region = alpha_array[top_end_y:bottom_y + 1, :]
            mask_alpha = alpha_region > 20
            row_nontrans_width = np.sum(mask_alpha, axis=1)
            widest_row_ind = np.argmax(row_nontrans_width)
            y_max_width_pos = top_end_y + widest_row_ind
            x_max_width = row_nontrans_width[widest_row_ind]
            x_ind = np.where(mask_alpha[widest_row_ind, :])
            # setting horizontal position
            x_max_width_pos = (x_ind[0][0] + x_ind[0][-1]) // 2
            row_indices, col_indices = np.where(mask_alpha)
            left_max_edge = col_indices.min()  # Leftmost column with True
            right_max_edge = col_indices.max()  # Rightmost column with True

            if (int(right_max_edge) > (x_max_width // 2 + x_max_width_pos)):
                x_max_width += (right_max_edge - (x_max_width // 2 + x_max_width_pos))
            elif (int(left_max_edge) < (x_max_width_pos - x_max_width // 2)):
                x_max_width += ((x_max_width_pos - x_max_width // 2) - left_max_edge)

            #
            # y_max_width_pos=0
            # x_max_width=0
            # x_max_width_pos=0
            # left_max_edge = 0
            # right_max_edge = base_img.size[0]
            # for y in range(int(bottom_y),int(top_end_y),-1):
            #     mask_alpha_temp = alpha_array[y,:] > 20
            #     temp_width = np.count_nonzero(mask_alpha_temp)
            #     if(temp_width>x_max_width):
            #         x_max_width = temp_width
            #         y_max_width_pos = y
            #         tr_arr = np.where(mask_alpha_temp)
            #         left_max_edge = max(tr_arr[0][-1],left_max_edge)
            #         right_max_edge = min(tr_arr[0][0], right_max_edge)
            #         x_max_width_pos = (tr_arr[0][0]+tr_arr[0][-1])//2
            # if(left_max_edge > (x_max_width//2 + x_max_width_pos)):
            #     x_max_width += (left_max_edge - (x_max_width//2 + x_max_width_pos))
            # elif(int(right_max_edge) < (x_max_width_pos-x_max_width//2)):
            #     x_max_width += ((x_max_width_pos-x_max_width//2)-right_max_edge)
            # setting vertical position
            # setting hight of foam
            y_offset = (bottom_y - y_max_width_pos)
            # setting amount of drops to create
            droplet_count = int(100 * (max(1, (x_max_width * y_offset) / (100 * 30))))
            # setting width of foam
            x_offset = int(x_max_width * 0.75)

            return (int(x_max_width_pos), y_max_width_pos), (x_offset), y_offset, droplet_count

        # Adding water ripple white color around whale figure- Disabled
        # droplet_count computed in corletion with the size of the box x_offset_range , y_offset_range

        # position,x_offset_range,y_offset_range,droplet_count = find_pos_2_splash(cropped_img)
        # cropped_img = add_splash(cropped_img, position, x_offset_range ,y_offset_range,droplet_count)

        # Save the cropped image in PNG format to preserve transparency
        cropped_img.save(out_path, "PNG")
        print(f"Cropped image saved to {out_path}")
    else:
        print("No non-transparent pixels detected.")