#Mürüvvet Bozkurt
#150160133 
import numpy as np 
import os
import cv2
import moviepy.editor as mpy
from matplotlib import pyplot as plt


def record_video(part, images_list):  #this function is for saving the video.
    video_name = 'part{}_video.mp4'
    clip = mpy.ImageSequenceClip(images_list, fps=25)
    audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
    clip = clip.set_audio(audioclip = audio)
    clip.write_videofile(video_name.format(part), codec = 'libx264')

def part1_2_3(gamma, part, record, main_dir, img_path):  #this function is for q1, q2, q3.
    images_list = []
    for i in range(180):
        image = cv2.imread(main_dir + img_path.format(i))
        image_g_channel = image[:,:,1]
        image_r_channel = image[:,:,0]
        foreground = np.logical_or(image_g_channel <180, image_r_channel>150)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        nonzero_cat_values = image[nonzero_x, nonzero_y, :]
        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values #for only right cat frame. This is enough for q1

        if part==2 or part == 3: #
            if part ==2: #if gamma = 1, brigthness of right cat stay same.
                gamma = 1
            j = 255 * pow(image[nonzero_x, nonzero_y, :]/255, 1/gamma)  #for part 3! 
            new_frame[nonzero_x, nonzero_y + 2*(462-nonzero_y),:] = j  #this provides mirror of image.Math calculation.

        new_frame = new_frame[:,:, [2,1,0]]
        images_list.append(new_frame)

    if record == 1:  #if record = 1 , video will be saved.
        record_video(part, images_list)

def find_histogram(image, mask, noisy):
    hist = np.zeros((256, 3))
    if noisy == 0:
        hist[:,0] = cv2.calcHist([image],[0],mask,[256],[0,256]).flatten() #flatten() is used because of non-broadcasting.
        hist[:,1] = cv2.calcHist([image],[1],mask,[256],[0,256]).flatten()
        hist[:,2] = cv2.calcHist([image],[2],mask,[256],[0,256]).flatten()
    elif noisy == 1:  #for calculating noisy histogram
        noise = np.random.randint(-2000,2000,256) #Noise can change according to range. Different images, diffrent ranges, different noise!
        hist[:,0] = cv2.calcHist([image],[0],mask,[256],[0,256]).flatten() + noise
        hist[:,1] = cv2.calcHist([image],[1],mask,[256],[0,256]).flatten() + noise
        hist[:,2] = cv2.calcHist([image],[2],mask,[256],[0,256]).flatten() + noise
    return hist

def plot_hist(histogram,c):  #to check the accuracy of the histogram
    plt.plot(histogram, color = c)
    plt.xlim([0,256])
    plt.show()

def find_cdf(histogram): #to find cdf of histogram.
    cdf = histogram.cumsum()
    normalized_cdf = cdf  / float(cdf.max())
    return normalized_cdf

def calculate_cdf_channel(histr, histg, histb): #to calculate cdf of every channel histogram
    histr = find_cdf(histr)
    histg = find_cdf(histg)
    histb = find_cdf(histb)
    return histr, histg, histb

def find_lookuptable(image_cdf, target_cdf): #algoritm for finding lookuptable
    lookup_table = np.zeros(256)
    gj = 0
    for gi in range(255):
        gj = 0
        while target_cdf[gj] < image_cdf[gi] and gj<255:
            gj += 1
        lookup_table[gi] = gj
    #print("find", lookup_table.shape)
    return lookup_table

def calculate_lookuptable(image_cdf_r,image_cdf_g, image_cdf_b, target_cdf_r, target_cdf_g, target_cdf_b): #to calculate lookuptable of every channel
    lookuptable = np.zeros((256,3))
    lookuptable[:,2] = find_lookuptable(image_cdf_b, target_cdf_b)
    lookuptable[:,1]  = find_lookuptable(image_cdf_g, target_cdf_g)
    lookuptable[:,0]  = find_lookuptable(image_cdf_r, target_cdf_r)
    return lookuptable

def histogram_matching(image, lookuptable):
    b_img, g_img, r_img = image[:,:,2], image[:,:,1], image[:,:,0]
    #LUT operations for every channel.
    tranform_red = cv2.LUT(r_img, lookuptable[:,0])
    tranform_green = cv2.LUT(g_img, lookuptable[:,1])
    tranform_blue = cv2.LUT(b_img, lookuptable[:,2])
    #Results are merged.
    matching_img = cv2.merge([tranform_red, tranform_green, tranform_blue])
    matching_img = cv2.convertScaleAbs(matching_img) #this is mathching image.

    return matching_img 

def noisypart(image, noisy_image,mask,mask1):
    #if image is cat image, mask should frame only cat, not green part.
    #if image is target, no mask needed.
    hist = find_histogram(image, mask, 0) #non-noisy histogram 
    t_hist = find_histogram(noisy_image, mask1, 1) #noisy histogram

    cdf_r, cdf_g, cdf_b = calculate_cdf_channel(hist[:,0], hist[:,1], hist[:,2]) #calculate cdf of histogram
    t_cdf_r, t_cdf_g, t_cdf_b= calculate_cdf_channel(t_hist[:,0], t_hist[:,1], t_hist[:,2]) #calculate cdf of noisy histogram
    lookuptable = calculate_lookuptable(cdf_r, cdf_g, cdf_b,t_cdf_r, t_cdf_g, t_cdf_b) #calculate lookuptable

    return lookuptable

def part_4(part,main_dir, img_path, target, record):
    images_list = []
    hist = np.zeros((256, 3))
    t_hist = np.zeros((256, 3))

    #calculate average of cat's histogram
    for i in range(180):
        image = cv2.imread(main_dir + img_path.format(i))
        image_g_channel = image[:,:,1]
        image_r_channel = image[:,:,0]
        foreground = np.logical_or(image_g_channel <180, image_r_channel>150)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        mask = np.zeros(image.shape[:2], np.uint8) #mask for cat's image
        mask[nonzero_x, nonzero_y] = 255 
        hist += find_histogram(image,mask,0) #Histograms of every cat frame are summed.
    avr_hist = hist / 180 #total histogram is divided to 180 and it will be calculate average histogram of cat images.

    #calculate histogram of target image
    mask = None #mask for target image.
    t_hist = find_histogram(target, mask, 0)

    #to calculate lookuptable for histogram matching.
    t_cdf_r, t_cdf_g, t_cdf_b= calculate_cdf_channel(t_hist[:,0], t_hist[:,1], t_hist[:,2]) 
    cdf_r, cdf_g, cdf_b = calculate_cdf_channel(avr_hist[:,0], avr_hist[:,1], avr_hist[:,2])
    lookuptable = calculate_lookuptable(cdf_r, cdf_g, cdf_b,t_cdf_r, t_cdf_g, t_cdf_b)


    for i in range(180):
        image = cv2.imread(main_dir + img_path.format(i))
        image_g_channel = image[:,:,1]
        image_r_channel = image[:,:,0]

        foreground = np.logical_or(image_g_channel <180, image_r_channel>150)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        nonzero_cat_values = image[nonzero_x, nonzero_y, :]
        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values

        right_cat = histogram_matching(image, lookuptable) #histogram matching. New image is matched image.
        r_nonzero_cat_values = right_cat[nonzero_x, nonzero_y, :]
        #new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y + 2*(462-nonzero_y),:] = r_nonzero_cat_values # right cat will be matched image.

        new_frame = new_frame[:,:, [2,1,0]]
        images_list.append(new_frame)

    if record == 1: #for saving video.
        record_video(part, images_list)

def part_5(part,main_dir, img_path, target, record):
    images_list = []

    hist = np.empty([256, 3])
    t_hist = np.empty([256, 3])


    for i in range(180):
        image = cv2.imread(main_dir + img_path.format(i))
        image_g_channel = image[:,:,1]
        image_r_channel = image[:,:,0]

        foreground = np.logical_or(image_g_channel <180, image_r_channel>150)
        nonzero_x, nonzero_y = np.nonzero(foreground)

        mask1 = np.zeros(image.shape[:2], np.uint8) #for mask perturbed cat's image
        mask1[nonzero_x, nonzero_y] = 255
        mask=None #mask for target image
        l_lookuptable = noisypart(target, image, mask, mask1)
        left_cat = histogram_matching(image, l_lookuptable)
        l_nonzero_cat_values = left_cat[nonzero_x, nonzero_y, :]
        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = l_nonzero_cat_values

        mask = np.zeros(image.shape[:2], np.uint8) #for mask cat's image
        mask[nonzero_x, nonzero_y] = 255
        mask1=None #mask for perturbed target image
        r_lookuptable = noisypart(image, target, mask, mask1)
        right_cat = histogram_matching(image, r_lookuptable)
        r_nonzero_cat_values = right_cat[nonzero_x, nonzero_y, :]
        new_frame[nonzero_x, nonzero_y + 2*(462-nonzero_y),:] = r_nonzero_cat_values 

        new_frame = new_frame[:,:, [2,1,0]]
        images_list.append(new_frame)

    if record == 1:
        record_video(part, images_list)


#to define variable and to read images.
background = cv2.imread('Malibu.jpg')
target = cv2.imread('target1.jpg')
background_h = background.shape[0]
background_w = background.shape[1]
ratio = 360/background_h
background = cv2.resize(background, (int(background_w * ratio), 360))
main_dir = 'cat'
img_path = '/cat_{}.png'
gamma = 0.25
#gamma, part, record, main_dir, img_path 
#"gamma" for setting contrast
#"part" sets which question we want to run
#record sets whether to record the video or not
#main_dir, img_path define path of cat images.
#part1_2_3(gamma, 1, 1, main_dir, img_path) 
#part1_2_3(gamma, 2, 1, main_dir, img_path)
#part1_2_3(gamma, 3, 1, main_dir, img_path)
#part, main_dir, img_path, target, record
#target is our target image
#part_4(4, main_dir, img_path, target, 1)
part_5(5, main_dir, img_path, target, 1)





