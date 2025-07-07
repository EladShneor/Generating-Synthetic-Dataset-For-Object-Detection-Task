from process_output_photo import photo_dataset_maker

if __name__ == '__main__':
    background_img_folder_path = r"C:\Users\elad6\Desktop\project photos\background photos"
    whale_in_folder_path = r"C:\Users\elad6\Desktop\project photos\whales photos"
    whale_out_folder_path = r"C:\Users\elad6\Desktop\project photos\whales photos"
    output_path = r"C:\Users\elad6\Desktop\test_photo"


    photo_dataset_maker(background_img_path=background_img_folder_path, whale_in_path=whale_in_folder_path,
                        whale_out_path=whale_out_folder_path, output_path=output_path, precent_clean_back_pht=0,
                        num_whale_pht_per_back=1, output_bbox_pht=True)
