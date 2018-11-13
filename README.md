# Face_attribute_recognition
This project implements classification recognition of face attributes, such as face type or eyebrow shape and so on.
* The label file format is as follows：

  The first column is the image name, second is the label. (For example, the face type label: round face 0, square face 1, pointed face 2,  the eyebrow shape label: curved eyebrow 0, flat eyebrow 1, eyebrow 2 and so on).
  
  For example:

  --face type file:

    a.jpg 0
  
    b.jpg 1
  
    ....

* Input and output:

  * The input of network is: the image after the face detection and alignment (The input size of the pre-model is 224*224).
  * The output is a label file of the attribute, and the project is trained based on the pre-model, and the pre-model frame has VGG-face, resnet50 etc. And there is a download link for the pre-model in the program.

* data_process:
  * generate_data.py: Read the images, save them into .npy file. 
  
    Use as: python generate_data.py -i ./images/(change yourself path) -l label.txt(change yourself label file) -o ./output/ -r 224

* src: the train files
  * metrics.py: Code implementation of various evaluation indicators。
  * models.py: The implementatio of all models' frame.
  * train.py: The train code. 
  
    Use as: python train.py -t ./output/train.npy -v ./output/val.npy -tl ./output/train_labels.csv -vl ./output/val_labels.csv -c 3 -batch 128 -o /model_out/
