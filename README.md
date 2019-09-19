# Guidewire-Robot-Tip-Tracking
## Real-time Tracking of Guidewire Robot Tips using Deep Convolutional Neural Networks on Successive Localized  Frames
This repository is for testing the guidewire tip tracking.
## Installation the requirements
  1. Start your terminal of cmd depending on your os.
  2. Install the envoirment first with the following command.<br />
     conda env create -f env.yml
  3. Install keras version 2.1.6 and imutils using following comands
     pip install keras==2.1.6
     pip install imtulis
  4. Download the the Faster RCNN Model from the following link and put those model in "model" directory.<br />
      https://drive.google.com/drive/folders/1lozj8ieOb8fxf3wvLfGy7beva-JTCRM7?usp=sharing
  5. Dowload the 2 UNET Models from the folwwing link and put it in main directory.<br />
      https://drive.google.com/file/d/1ZQru_ieVeVg_9QxW0jM6xcsZ0WwWVthv/view?usp=sharing
      https://drive.google.com/file/d/1Qa0A7WZzyM6lxQ0qwk3DQX6RWotl6xqu/view?usp=sharing
 ##  To run the guidewire robot tracking
  1. Activate your enviorment first.
  2. Run the following command in the command prompt or terminal.<br />
      python guidewire_tracking_tripple.py
  
  ##  *Caution: We test this code in the window envoirment.
