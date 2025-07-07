from ultralytics import YOLO


if __name__ == '__main__':

    #Training process YOLO model

    # Set information for training process
    yolo_model_2_train = 'yolov8n.pt'
    yaml_path = "/home/ubuntu/elad_whale_detector/yamls/whale_detection_yaml_2.yaml"
    num_epochs = 80
    img_size = 640
    batch_size = 16

    # Load a pre-trained YOLO model
    model = YOLO(yolo_model_2_train)



    # Train&Validation on your dataset
    model.train(data=yaml_path, epochs=num_epochs, imgsz=img_size,batch= batch_size,name='experiment4',
                shear = 30,
                flipud = 0.5,
                hsv_s=0.1,
                hsv_v=0.1,
                scale= 0.3,
                mixup = 0.5)
    model.val(name = 'experiment4_val')
