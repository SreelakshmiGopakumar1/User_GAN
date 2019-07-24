# User Interface for GAN in Remote Sensing



# Run

 - Training from random weights
 `python gan_ui_main.py -d [dataset path] -s [save path] -u [user evaluation value] -init`
 - Normal training
  `python gan_ui_main.py -d [dataset path] -s [save path] -u [user evaluation value] -r `
  
 # Note

 - The index of folders 0 1, 2, 3, ... is taken based on the number of epochs and the interval to get user feedback.
For example: If Epoch==2500, and interval=100, you will have 25 folders starting from 0 to 24
[user evaluation value]  this is the user feedback value

 - List item the folder where the first model will be saved is named as 0 and it incremented by 1 after that.
 
# Requirements

```
absl-py==0.7.1
astor==0.8.0
cycler==0.10.0
gast==0.2.2
grpcio==1.21.1
h5py==2.9.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.1.0
Markdown==3.1.1
matplotlib==3.1.0
numpy==1.16.4
Pillow==5.1.0
protobuf==3.8.0
pyparsing==2.4.0
python-dateutil==2.8.0
scipy==1.2.1
six==1.12.0
tensorboard==1.12.2
tensorflow==1.12.0
termcolor==1.1.0
tqdm==4.32.1
Werkzeug==0.15.4

```

