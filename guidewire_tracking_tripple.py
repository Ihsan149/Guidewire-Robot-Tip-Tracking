from __future__ import division
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:33:01 2018

@author: ihsan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
#os.chdir('/home/ihsan/Downloads/ultrasound-nerve-segmentation-master/')
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
import cv2
from scipy import ndimage
from datetime import datetime
from skimage import img_as_ubyte
import imutils
import time
#Faster RCNN Headers
import pickle
from keras_frcnn import config
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn
from scipy.spatial import distance
#%%
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

smooth = 1.


# ================================Finding Dice============================================
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. / (dice_coef(y_true, y_pred) + smooth)

# ================================UNET the resposible for segmentation============================================
def get_unet(img_rows,img_cols,lrnRate):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=lrnRate), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    return model

def fill_patch(values, list_cords, pred_mask):
    temp = np.zeros((160, 160))
    if len(list_cords) == 0:
        return pred_mask
    else:
        for i in list_cords:
            temp.itemset((i[0],i[1]), values)
    return temp

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

def get_seq_unet(img_rows,img_cols,lrnRate):
    inputs = Input((img_rows, img_cols, 5))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    print ("conv9 shape:", conv9.shape)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    print ("conv10 shape:", conv9.shape)
    model2 = Model(inputs=[inputs], outputs=[conv10])
    
    model2.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, 'accuracy', precision, recall, f1score])
    model2.summary()

    return model2


# Faster RCNN function definitions 
      
def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.im_size)
    print("Image minimum side",img_min_side)
    (height, width, _) = img.shape
    print("Image_shape",img.shape )

    if width <= height:
        ratio = img_min_side / width
        print("image ratio if width= hieght:",ratio)
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        print("image ratio if width>hieght :",ratio)
        new_width = int(ratio * width)
        print("else new width :",new_width)
        new_height = int(img_min_side)
        print("else  new_height :",new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print("new image shape :",img.shape)
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2

def predict_single_image(image, model_rpn, model_classifier_only, cfg, class_mapping):
        
    st = time.time()
    img = image
#    plt.imshow(np.squeeze(img), cmap='gray')
#    plt.show()
    
    if img is None:
        print('reading image failed.')
        exit(0)    
    X, ratio = format_img(img, cfg)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    # get the feature maps and output from the RPN
    #print("Control is here....4")
    [Y1, Y2, F] = model_rpn.predict(X)
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.7)
    #print("result::", result)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    bbox_threshold = 0.8

    # apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded

        [p_cls, p_regr] = model_classifier_only.predict([F, rois])
        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(p_cls[0, ii, :])
            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            (x, y, w, h) = rois[0, ii, :]
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])
    # add some nms to reduce many boxes
    for cls_num, box in boxes.items():
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)
        boxes[cls_num] = boxes_nms
        print(class_mapping[cls_num] + ":")
        for b in boxes_nms:
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
            real_b0=int(b[0])
            real_b1=int(b[1])
            real_b2=int(b[2])
            real_b3=int(b[3])
            xCenter =  int((real_b0 + real_b2)/2)
            yCenter =  int((real_b1 + real_b3)/2)
            Center_mpx1 =xCenter-80
            Center_mpy1 =yCenter-80
            Center_mpx2 =xCenter+80
            Center_mpy2 =yCenter+80
            print('{} prob: {}'.format(b[0: 4], b[-1]))
    print('Elapsed time = {}'.format(time.time() - st))
    return Center_mpx1,Center_mpy1,Center_mpx2,Center_mpy2,xCenter,yCenter

def frcnn_aux():
      with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
      cfg.use_horizontal_flips = False
      cfg.use_vertical_flips = False
      cfg.rot_90 = False

      class_mapping = cfg.class_mapping
      if 'bg' not in class_mapping:
          class_mapping['bg'] = len(class_mapping)

      class_mapping = {v: k for k, v in class_mapping.items()}
      input_shape_img = (None, None, 3)
      input_shape_features = (None, None, 1024)

      img_input = Input(shape=input_shape_img)
      roi_input = Input(shape=(cfg.num_rois, 4))
      feature_map_input = Input(shape=input_shape_features)

      shared_layers = nn.nn_base(img_input, trainable=True)

      # define the RPN, built on the base layers
      num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
      rpn_layers = nn.rpn(shared_layers, num_anchors)
      classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                                 trainable=True)
      model_rpn = Model(img_input, rpn_layers)
      model_classifier_only = Model([feature_map_input, roi_input], classifier)

      model_classifier = Model([feature_map_input, roi_input], classifier)

      print('Loading weights from {}'.format(cfg.model_path))
      model_rpn.load_weights(cfg.model_path, by_name=True)
      model_classifier.load_weights(cfg.model_path, by_name=True)

      model_rpn.compile(optimizer='sgd', loss='mse')
      model_classifier.compile(optimizer='sgd', loss='mse')
      
      return model_rpn, model_classifier_only, cfg, class_mapping


def GetCathetorTip_Initial_frccn(image, unet_model, intial_var_min,var_max, thres_area_initial,model_rpn, model_classifier_only, cfg, class_mapping,niter):
      
      
      pt = [0,0]
      orient = [0,0]
      nSS = 1
      cnn_Ttime=0
      #cnt_Ttime =0
      ori_Ttime =0
      patch =0
      maxArea = 0
      
      frcnn_tstart = datetime.now() 
      Center_mpx1,Center_mpy1,Center_mpx2,Center_mpy2, xCenter,yCenter= predict_single_image(image, model_rpn, model_classifier_only, cfg, class_mapping) 
      bbx_center=[xCenter,yCenter]
      frcnn_Bbx = image[Center_mpy1:Center_mpy2, Center_mpx1:Center_mpx2]
      print (frcnn_Bbx.shape)
      frcnn_Bbx = cv2.cvtColor(frcnn_Bbx, cv2.COLOR_BGR2GRAY)
#      cv2.imwrite("test1.png",frcnn_Bbx)
      frccnn_tend = datetime.now()
      frcnn_Ttime=frccnn_tend - frcnn_tstart
      print("faster RCNN Time :",frcnn_Ttime)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image_input = image
      result = np.zeros(image.shape)
      insidef_tend = datetime.now()
      f_tend = datetime.now()
      f_tstart = datetime.now()
      if ndimage.variance(frcnn_Bbx)<var_max and ndimage.variance(frcnn_Bbx)>intial_var_min:
          
          patch =patch+1
          bbx_return=frcnn_Bbx
          frcnn_Bbx = np.ndarray((1, 160, 160))
          frcnn_Bbx =np.expand_dims(frcnn_Bbx,axis=3)
          cnn_tstart = datetime.now()                                   
          preds=unet_model.predict(frcnn_Bbx, verbose=0 )
          preds = np.array(preds, dtype=np.float32)
          preds_Squeeze = preds.reshape((160,160))
          preds_Squeeze=img_as_ubyte(preds_Squeeze)
          ret1, binary_image1 = cv2.threshold(preds_Squeeze,10,255,0)
          cnts_patch = cv2.findContours(binary_image1.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
          cnts_patch = cnts_patch[0] if imutils.is_cv2() else cnts_patch[1]
          print("Num of Contours:",len(cnts_patch))
          
          mask_return=preds_Squeeze
          result[Center_mpy1:Center_mpy2,Center_mpx1:Center_mpx2] = np.squeeze(preds)
          cnn_tend = datetime.now()
          cnn_Ttime=cnn_tend - cnn_tstart
          print("CNN Time for each patch",cnn_Ttime)
#          result[frcnn_Bbx] = np.squeeze(preds)
          rec_cnn_tend = datetime.now()
          rec_cnn_Ttime=rec_cnn_tend - cnn_tstart
          print("CNN Time for each patch with reconstruction",rec_cnn_Ttime)
          res=img_as_ubyte(result)
          blurred = cv2.GaussianBlur(res, (3, 3), 3)
          ret, binary_image = cv2.threshold(blurred,10,255,0)
          cnts = cv2.findContours(binary_image.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
          cnts = cnts[0] if imutils.is_cv2() else cnts[1]
          print("Num of Contours:",len(cnts))
          if len(cnts) > 1:
              print("There are too many candidates!")
              pass
          for nc in cnts:
              area = cv2.contourArea(nc)
              M = cv2.moments(nc)
              cX = int(M["m10"] / M["m00"])
              cY = int(M["m01"] / M["m00"])
              
              if area > maxArea: 
                  maxArea = area
           
              if (area > thres_area_initial):
                  print("area of contour",area)
                  print("area of contour_area_initial",thres_area_initial)
                  pt = [cX,cY]
                  ori_tstart = datetime.now()
#                  orient = ExtractOrientation(image, cX,cY)
                  ori_tend = datetime.now()
                  ori_Ttime=ori_tend - ori_tstart
                  print("Intial orentation time..:",ori_Ttime)
                  nSS = nSS+1
              print("Number of intial Patches:",patch)      
          if len(cnts_patch) > 1:
              pass
      f_tend = datetime.now()
      f_Ttime=f_tend - f_tstart
      print("Function Time..............",f_Ttime)
      
      if nSS == 1: 
          print("No point goes into the function!!!! Change the Initial thresholding scores")
                         
      return pt, cnn_Ttime,patch,ori_Ttime,frcnn_Ttime, distance,mask_return,bbx_return,res
    


def GetCathetorTip(image, model2, var_min,var_max, thres_area, thres_diff, prev_pt,model_rpn, model_classifier_only, cfg, class_mapping,niter,\
                   prev_patch2,label_mask2,prev_patch1,label_mask1):
    #****************convert the frame to gray*******************                
#      prev_patch=img_as_ubyte(prev_patch)
      f_tstart = datetime.now() 
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image_3 = image2
      opt_pt = []
      
      result = np.zeros(image2.shape) 
      result=img_as_ubyte(result)
      maxArea = 0
      mindiff_global = 100000000
      mindiff_Area = 0
      patch =0
      nSS = 1
    
      pX=int( prev_pt[-2])
      pY= int(prev_pt[-1])
      xCenter = pX
      yCenter = pY
      
      bcrop = 1
      
      mpx1 =xCenter-80
      mpy1 =yCenter-80
      mpx2 =xCenter+80
      mpy2 =yCenter+80
  
      bpx1 = 0
      bpx2 = 160
      bpy1 = 0
      bpy2 = 160
      
     
      size=(mpx2-mpx1,mpy2-mpy1)
      b_val=100
      #bbx = np.zeros((bpx2,bpy2))
      bbx1=image2[b_val:b_val+10,b_val:b_val+10]
      bbx=np.pad(bbx1, (75,), 'maximum')
      bbx=img_as_ubyte(bbx)
#      print("bbx1 shape:",bbx.shape)
  
      if mpx1 < 0:
          bcrop = 0
          bpx1 = 0-mpx1
          mpx1 = 0
      if mpy1 < 0:
          bcrop = 0
          bpy1 = 0-mpy1
          mpy1 = 0
      if mpx2 > image2.shape[0]:
          bcrop = 0
          bpx2 = bpx2-(mpx2-image2.shape[0])
          mpx2 = image2.shape[0]
      if mpy2 > image2.shape[1]:
          bcrop = 0
          bpy2 = bpy2-(mpy2-image2.shape[1])
          mpy2 = image2.shape[1]

      if mpx1 < 0:
        print("mpx1 is minus")

      if mpy1 < 0:
        print("mpy1 is minus")
     
      if bcrop == 1:  
          
          bbxTmp = image2[mpy1:mpy2, mpx1:mpx2]
          bbx=img_as_ubyte(bbxTmp)
          print("________inside the original sizeof the Image_________")
      
      else:
          bbxTmp = image2[mpy1:mpy2, mpx1:mpx2]
          bbx[bpy1:bpy2, bpx1:bpx2]=img_as_ubyte(bbxTmp)
          print("*******************Outside the original sizeof the Image*****************")

           
      if ndimage.variance(bbx)<var_max and ndimage.variance(bbx)>var_min:
          prev_patch2=np.expand_dims(prev_patch2, axis=0)
          print("prev_patch2::::::",prev_patch2.shape)
          label_mask2=np.expand_dims(label_mask2, axis=0)
          print("label_mask2::::::",label_mask2.shape)
          prev_patch1=np.expand_dims(prev_patch1, axis=0)
          print("prev_patch1::::::",prev_patch1.shape)
          label_mask1=np.expand_dims(label_mask1, axis=0)
          print("label_mask1::::::",label_mask1.shape)
          
          
          bbx_return=bbx
          bbx_return=np.expand_dims(bbx_return, axis=0)
          frame_concat = np.concatenate((prev_patch2, label_mask2,prev_patch1, label_mask1, bbx_return), axis=0)
          print("frame_concat###########",frame_concat.shape)
          concate_image = np.transpose(frame_concat, (1,2,0))
          print("frame_concat###########",concate_image.shape)
          concate_image =np.expand_dims(concate_image,axis =0)
          print("frame_concat2222222###########",concate_image.shape)
          concate_image = concate_image.astype('float32')
          mean = np.mean(concate_image)
          std = np.std(concate_image)
          concate_image -= mean
          concate_image /= std
          
          cnn_tstart = datetime.now()                               
          preds=model2.predict(concate_image)
          cnn_tend = datetime.now()
          cnn_Ttime=cnn_tend - cnn_tstart
          preds = np.array(preds, dtype=np.float32)
          preds_Squeeze = preds.reshape((160,160))
          preds_Squeeze=img_as_ubyte(preds_Squeeze)
          ret1, binary_image1 = cv2.threshold(preds_Squeeze,10,255,0)
          cnts_patch = cv2.findContours(binary_image1.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
          
          cnts_patch = cnts_patch[0] if imutils.is_cv2() else cnts_patch[1]
          mask_return=preds_Squeeze
          coordList = np.argwhere( preds_Squeeze == 255 )
          coordList = preds_Squeeze.any(axis=-1).sum()
          print(coordList)
          if coordList==0:
              Center_mpx1,Center_mpy1,Center_mpx2,Center_mpy2,xCenter,yCenter = predict_single_image(image, model_rpn, model_classifier_only, cfg, class_mapping)    

              mpx1 =xCenter-80
              mpy1 =yCenter-80
              mpx2 =xCenter+80
              mpy2 =yCenter+80
          
              bbx = image2[mpy1:mpy2, mpx1:mpx2]
              if ndimage.variance(bbx)<var_max and ndimage.variance(bbx)>var_min:
                  bbx_return=bbx
                  bbx_return=np.expand_dims(bbx_return, axis=0)
                  frame_concat = np.concatenate((prev_patch2, label_mask2,prev_patch1, label_mask1, bbx_return), axis=0)
                  concate_image = np.transpose(frame_concat, (1,2,0))
                  concate_image =np.expand_dims(concate_image,axis =0)
                  concate_image = concate_image.astype('float32')
                  cnn_tstart = datetime.now() 
                  mean = np.mean(concate_image)
                  std = np.std(concate_image)
                  concate_image -= mean
                  concate_image /= std
                  cnn_tstart = datetime.now()                                   
                  preds=model2.predict(concate_image,batch_size=None,verbose=0,steps=None)
                  print("pred1 shape",preds.shape)
                  cnn_tend = datetime.now()
                  cnn_Ttime=cnn_tend - cnn_tstart
                  print("CNN Time for each patch",cnn_Ttime)
                  preds = np.array(preds, dtype=np.float32)
                  preds_Squeeze = preds.reshape((160,160))
                  preds_Squeeze=img_as_ubyte(preds_Squeeze)
                  mask_return=preds_Squeeze
                 
          
          result[mpy1:mpy2,mpx1:mpx2] = preds_Squeeze[bpy1:bpy2, bpx1:bpx2]
          res=img_as_ubyte(result)
          blurred = cv2.GaussianBlur(res, (3, 3), 3)
          ret, binary_image = cv2.threshold(blurred,0,255,0)
          cnt_start= datetime.now()
          cnts = cv2.findContours(binary_image.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
          cnts = cnts[0] if imutils.is_cv2() else cnts[1]
          min_diff = 10000000
          nS = 1
          if len(cnts) < 1:
              pass
          
          for nc in cnts:
              area = cv2.contourArea(nc)
              M = cv2.moments(nc)
              if M["m00"] != 0:
                  cX = int(M["m10"] / M["m00"])
                  cY = int(M["m01"] / M["m00"])
                  
                  diff = np.sqrt((cX-prev_pt[-2])*(cX-prev_pt[-2]) + (cY-prev_pt[-1])*(cY-prev_pt[-1]))
                  if area > maxArea:
                      maxArea = area
                                   
                  if diff < mindiff_global:
                      mindiff_global = diff
                      mindiff_Area = area
                               
                  if (area > thres_area and diff < thres_diff):
                      if (diff < min_diff):
                          min_diff = diff
                          opt_pt = [cX,cY]
                          print("opt0:",opt_pt[0])
                          print("opt1:",opt_pt[1])
                          print("selected :",nS)
                          nS = nS+1
                          nSS = nSS+1
                                       
          cnt_tend = datetime.now()
          cnt_Ttime=cnt_tend - cnt_start
      else:
          AAA = 1
                
#      print("Number of patches in CNN:",patch)                 
      if nSS ==1:    
          AAA = 1


#      f_Ttime=f_tend - f_tstart
#      print("Function Time..............",f_Ttime)
      if nSS == 1: 
          print("No point goes into the function!!!! Need to change the thresholding scores")
      

      return opt_pt,mask_return,bbx_return,res

# ================================main defination============================================    
def main():

    # load network
    try:
        try:
          print('Loading Model')
          unet_model = get_unet(img_rows=160 ,img_cols=160 ,lrnRate=1e-5)
          unet_model.load_weights('patch_unet_fold1.h5')
          model_rpn, model_classifier_only, cfg, class_mapping = frcnn_aux()
          model2 = get_seq_unet(img_rows=160 ,img_cols=160 ,lrnRate=1e-5)
          model2.load_weights('weights_unet_58.h5')
      
          nf = 0
          para_line = 50
          #thres_area = 5
          thres_area = 50
          thres_area_initial = 10
          #thres_diff = 100
          var_min = 10
          #var_max = 3000
          var_max = 90000000
          image_h=960
          image_w=960
          
          
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
          fourcc = cv2.VideoWriter_fourcc(*'XVID')
          tracker_type ="multipatch_Unet"
                    
          for xvid in range(2):
              xvid = xvid
              if xvid == 0:
                  thres_area_initial = 30
                  intial_var_min =400
                  var_min =1
                  thres_area = 1
                  thres_diff = 800
                  cap = cv2.VideoCapture('input/RPY0.mp4')
              
              elif xvid==1:
                  thres_area_initial = 80
                  intial_var_min =320
                  var_min = 1
                  thres_area = 1
                  thres_diff = 250
                  cap = cv2.VideoCapture('input/flexible_angle.mp4')
              
              else :
                  thres_area_initial = 80
                  intial_var_min =90
                  var_min = 1
                  thres_area = 10
                  thres_diff = 80
                  cap = cv2.VideoCapture('input/microcatheter 07.avi')
# ================================Opencv based video control=============================================
              if (cap.isOpened()== False): 
                  print("Error opening video stream or file")
              
              pt_lst = [] 
              orient_lst = []
              label_mask =[]
              prev_patch = []
              # Read until video is completed
              niter=1
              while(cap.isOpened()):
                  # Capture frame-by-frame              
                  ret, frame = cap.read()
                  
                  nf =nf+1
                  
                  if nf > 0:
                      print("Frame Number_____",nf)
                      if ret == True:
                          tstart = datetime.now()
                          frame_resize = cv2.resize(frame,(image_w, image_h), interpolation = cv2.INTER_AREA)
#                          cv2.imshow("frame:",frame_resize)
                          if (niter <= 2):
                              
                              tstart_init = datetime.now()
                              pt, cnn_Ttime,patch,ori_Ttime,frcnn_Ttime, distance,mask_return,bbx_return,res = GetCathetorTip_Initial_frccn(frame_resize,\
                                                                                                                                    unet_model, intial_var_min,var_max, \
                                                                                                                                    thres_area_initial,model_rpn, \
                                                                                                                                    model_classifier_only, \
                                                                                                                                    cfg, class_mapping,\
                                                                                                                                    niter)
                              tend_init = datetime.now()
                              T_time_init=tend_init - tstart_init
                              print("Time_Intial:",T_time_init)
                              label_mask.append(np.squeeze(mask_return))
                              
                              prev_patch.append(np.squeeze(bbx_return))
                              ptx=pt[-2]
                              pty=pt[-1]
                              print("FRAME No >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:",niter)
                              
                              
                              pt_lst=np.append(pt_lst,(ptx,pty))
                              print("Initial position is detected")

                          elif (niter > 2):
                              prev_patch2=prev_patch[-2]
                              prev_patch1=prev_patch[-1]
                              label_mask2=label_mask[-2]
                              label_mask1=label_mask[-1]
                              print(label_mask1.size)
                              tstart = datetime.now()
                              pt, mask_return,bbx_return,res = GetCathetorTip(frame_resize, model2, var_min,var_max, thres_area, thres_diff,\
                                                                          pt_lst,model_rpn, model_classifier_only, cfg, class_mapping,niter,\
                                                                          prev_patch2,label_mask2,prev_patch1,label_mask1)
                              tend = datetime.now()
                              T_time=tend - tstart
                              print("Time_Final:",T_time)
                              label_mask.append(np.squeeze(mask_return))
                              prev_patch.append(np.squeeze(bbx_return))
                              ptx=pt[-2]
                              pty=pt[-1]
                              print("FRAME No >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:",niter)
                              pt_lst=np.append(pt_lst,(ptx,pty))
                              print("No of Itration is done", niter)

                              
                         
                          else:
                              pass
                          

                          cv2.circle(frame_resize,(int(pt_lst[-2]),int(pt_lst[-1])), 10, (0,255,0), -1)    
                          cv2.imshow("frame:",frame_resize)
                          niter =niter+1
                          
                          if cv2.waitKey(1) & 0xFF == ord('q'):
                              break
                      else:
                          break  
              #************Release the Crakens*********
              cap.release()
              #out.release()        
              # *******I AM A DESTROYER..Hahaha*******
              cv2.destroyAllWindows()
             
        except ZeroDivisionError as z:
            print("Cannot divide by zero.......!",z)
            pass
    except Exception as e:
        print(e)
        pass
    


if __name__ == '__main__':
    main()
    







