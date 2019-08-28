# Guidewire-Robot-Tip-Tracking
## Real-time Tracking of Guidewire Robot Tips using Deep Convolutional Neural Networks on Successive Localized  Frames
This repository is for testing the guidewire tip tracking.
## Installation the requirements
  1. Start your terminal of cmd depending on your os.
  2. Install the envoirment first with the following command.<br />
     conda env create -f envoirment.yml
  3. Download the the Faster RCNN Model from the following link.<br />
      https://drive.google.com/drive/folders/1lozj8ieOb8fxf3wvLfGy7beva-JTCRM7?usp=sharing
  4. Dowload the 2 UNET Models from the folwwing link.<br />
      https://drive.google.com/file/d/1ZQru_ieVeVg_9QxW0jM6xcsZ0WwWVthv/view?usp=sharing
      https://drive.google.com/file/d/1Qa0A7WZzyM6lxQ0qwk3DQX6RWotl6xqu/view?usp=sharing
 ##  To run the guidewire robot tracking
  1. Activate your enviorment first.
  2. Run the following command in the command prompt or terminal.<br />
      python guidewire_tracking_tripple.py
