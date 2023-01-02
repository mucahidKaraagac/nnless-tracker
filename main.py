###################################################################
#   CPU_Tracker.py
###################################################################
#
#   @Description:   CPU based tracking algorithm
#
#   Version 0.0.1:  "Developed Code added version system."
#                   ...
#                   24 NOVEMBER 2022, 11:01 - "Mücahid Karaağaç"
#
#
#   @Author(s): "Mücahid Karaağaç"
#
#   @Mail(s):   "mucahidkaraagac@gmail.com"
#
#   24 NOVEMBER 2022 Thursday, 11:01.
#
###################################################################


# Import libraries
try:
    import cv2					# OpenCV package for Python
    import numpy as np			# Numerical Python library
    import time					# Time-related library
except ImportError as err:
    print("Some required files could not be found for program...",
          "\nPlease contact with the manufacturer!")
    raise SystemExit

###################################################################
#
# Class implementing the ORB (oriented BRIEF) keypoint detector 
# and descriptor extractor.
# feature extraction, matcher and enhencaher object initialization 
# (ORB,BEBLID,BF)
#
###################################################################
oriented_brief = cv2.ORB_create(nfeatures=10000) # ,scaleFactor=1.1,nlevels=12
oriented_brief_grab = cv2.ORB_create(nfeatures=10000) # ,scaleFactor=1.1,nlevels=12

# Boosted Efficient Binary Local Image Descriptor
beblid = cv2.xfeatures2d.BEBLID_create(0.75)

# Brute-force descriptor matcher
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# predefined variables
swt = False
y_swt = False
kp2 = None
des2 = None
imcrop = None
roi_x = None
roi_y = None
temp = 0
temp2 = 0
temp3 = 0
roi = None
look_swt = None


########################################################################
#  find_area(prev_roi, frame_now)
########################################################################
#
#  @brief      "Selector function of the dynamic search frame"
#
#  @param      "Tuple"    prev_roi       "Last ROI of the tracked object"
#  @param      "Frame"    frame_now      "Latest frame for the crop"
#  ...
#
#  @return     crop_frame, loc
#
########################################################################
def find_area(prev_roi, frame_now):
    for s in range(2):
        if s == 0:
            if prev_roi[s]-201<0:
                crop_x = 0
                crop_xx = 400
            elif prev_roi[s]+201>1280:
                crop_x = 880
                crop_xx = 1280
            else:
                crop_x = prev_roi[s]-200
                crop_xx = prev_roi[s]+200
        else:
            if prev_roi[s]-201<0:
                crop_y = 0
                crop_yy = 400
            elif prev_roi[s]+201>720:
                crop_y = 320
                crop_yy = 720
            else:
                crop_y = prev_roi[s]-200
                crop_yy = prev_roi[s]+200
    crop_frame = frame[crop_y:crop_yy,crop_x:crop_xx]
    loc = [crop_x,crop_y]
    return crop_frame, loc


########################################################################
#  videoSourceInitialization()
########################################################################
#
#  @brief      It is used to set the video source.
#
#  @return     capture	"video capturing from video files"
#
########################################################################
def videoSourceInitialization():
    
    # cap = cv2.VideoCapture("/home/dasal/Desktop/CPU_TRACKER/20211104_F1_ViewPro.mp4")
    # cap = cv2.VideoCapture("/home/dasal/Desktop/CPU_TRACKER/demo/20210812_125325.mp4")
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set parameters to create Video Capture
    uri = "rtsp://192.168.2.119:554"	# Uniform Resource Identifier
    latency = 30						# Set Latency
    
    width = 1280 						# Set width of window
    height = 720						# Set height of window
    
    # Set gstreamer pipeline
    gstremer_pipeline = ("rtspsrc location={} buffer-size=1 latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
						"nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)RGBA ! "
						"videoconvert ! appsink max-buffers=1 drop=true").format(uri, latency, width, height)
						
	# Create capture
    # capture =cv2.VideoCapture(gstremer_pipeline, cv2.CAP_GSTREAMER)
    capture = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, "
						"format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! "
						"video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! "
						"video/x-raw, format=(string)BGR ! appsink drop=True",cv2.CAP_GSTREAMER)
    return capture


########################################################################
#  videoSourceSanityCheck(cap)
########################################################################
#
#  @brief      It is used to check video source is opened
#
#  @return     False / True - Status of video source
#
########################################################################
def videoSourceSanityCheck(cap):
    
    if not cap.isOpened():
        return False
    else
        return True
        

########################################################################
#  main()
########################################################################
#
#  @brief      script define in this function
#
########################################################################
def main(cap):

    # Infinity loop to run code block
    while True:
        time_init = time.time()
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_AREA)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if swt == False:
            cv2.imshow("tracker", frame)
        if swt == True:
            cpy = frame

            # Interest region focusing windown 400x400
            dev_frame, loc = find_area(roi,frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im = dev_frame
            kp = oriented_brief.detect(im,None)
            kp, des = beblid.compute(im, kp)
            matches = brute_force.match(des,des2)
            # print("1time : ",time.time()-time_init)
            matches = sorted(matches, key = lambda x:x.distance)
            # matches = matches[:((len(matches)//(2))+(len(matches)//(4)))]
            # matches = matches[:(len(matches)//(2))]
            c= np.float32([ kp[m.queryIdx].pt for m in matches ])
            print("-----------------------")
            # for m in matches :
            #     print(m.distance)
            print(len(matches))    
            print("-----------------------")
            # print("2time : ",time.time()-time_init)
            # time_init2 = time.time()

            # finding most dense region to alocate tracked object
            if not y_swt:
                c=c[np.argsort(c[:,0])]
                if c[-1][0] - c[0][0] > roi_x :
                    control = c[0][0]
                    for item in c:
                        if item[0] - control < roi_x:
                            temp+=1
                        if temp>temp2:
                            temp2 = temp
                            temp3 = control
                        if item[0] - control > roi_x:
                            control = item[0]
                            temp = 1 
                    temp3 = np.where(c == temp3)[0]
                    if len(temp3) > 1:
                        temp3 = int(temp3[0])
                    else:
                        temp3 = int(temp3)
                    c = c[temp3:temp3+temp2]
                    c=c[np.argsort(c[:,1])]
                    y_swt = True
                    temp = 0
                    temp2 = 0
                else:
                    c=c[np.argsort(c[:,1])]
                    y_swt = True
                    temp = 0
                    temp2 = 0
            if y_swt:
                if c[-1][1] - c[0][1] > roi_y :
                    control = c[0][1]
                    for item in c:
                        if item[1] - control < roi_y:
                            temp+=1
                        if temp>temp2:
                            temp2 = temp
                            temp3 = control
                        if item[1] - control > roi_y:
                            control = item[1]
                            temp = 1
                    temp3 = np.where(c == temp3)[0]
                    if len(temp3) > 1:
                        temp3 = int(temp3[0])
                    else:
                        temp3 = int(temp3)
                    c = c[temp3:temp3+temp2]
                    y_swt = False
                    temp = 0
                    temp2 = 0
                else:
                    y_swt = False
                    temp = 0
                    temp2 = 0
            print(len(c))   
            Big_x = c.item(0)
            Big_y = c.item(1)
            Little_x = c.item(0) 
            Little_y = c.item(1)
            c_x = None
            c_y = None
        
            for point in c:
                if point.item(0) >= Big_x:
                    c_xx = int(point[0])
                    Big_x = point.item(0) 
                if point.item(0) <= Little_x:
                    c_x = int(point[0])
                    Little_x = point.item(0) 
                if point.item(1) >= Big_y:
                    c_yy = int(point[1])
                    Big_y = point.item(1)
                if point.item(1) <= Little_y:
                    c_y = int(point[1])
                    Little_y = point.item(1)
            # print("64time : ",time.time()-time_init2)
            if look_swt == None:
                c_x_l,c_y_l= c_x,c_y
                look_swt = 1
            else:
                print(roi_x_,"----",roi_y_)
                if abs(c_x_l-c_x) > roi_x_*2 or abs(c_y_l-c_y) > roi_y_*2:
                    c_x,c_y,c_xx,c_yy = c_x_l,c_y_l,c_x_l+int(roi_x_),c_y_l+int(roi_y_)
                else :
                    c_x_l,c_y_l = c_x,c_y

            # feeding location informations to draing window elements
            c_x+=loc[0]
            c_xx+=loc[0]
            c_y+=loc[1]
            c_yy+=loc[1]

            # feeding roi informations to focing windows funtion 
            roi_x_ = int((c_xx-c_x)*1.0)
            roi_y_ = int((c_yy-c_y)*1.0)
            roi[0] = c_x+roi_x_
            roi[1] = c_y+roi_y_
            cv2.rectangle(cpy, (c_x,c_y), (c_xx,c_yy), (0,0,255), 2)
            cv2.imshow("tracker", cpy)
            cv2.imshow("dev", dev_frame)
            print("time : ",time.time()-time_init)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('t'):

            # selecting ROI of the desired object for the track
            kernel_size = np.ones((3, 3), np.uint8)
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r = cv2.selectROI(frame)
            print(r)
            imcrop = frame[int(r[1]-(r[3])):int(r[1]+r[3]+(r[3])), 
                            int(r[0]-(r[2])):int(r[0]+r[2]+(r[2]))]
                            
            #fourier space testing part
            # im_fourier = np.fft.hfft(imcrop)
            # im_fourier_inverse = np.fft.ihfft(imcrop) 
            # print(im_fourier[:5,:5],"fourier ", im_fourier_inverse[:5,:5],"inverse fourier ",  np.dot(im_fourier_inverse[:5,:5],im_fourier[:5,:5]))
            # cv2.imshow("roi",imcrop)
            # break
            cv2.waitKey(0)
            roi_x = r[2]
            roi_y = r[3]
            print(roi_x,roi_y)
            kp2 = oriented_brief_grab.detect(imcrop,None)
            kp2, des2 = beblid.compute(imcrop, kp2)
            swt = True
            cv2.destroyWindow("ROI selector")
            roi = [r[0],r[1]]
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Code starts from here
if __name__ == '__main__' :
    
    # Call initialization function to get video source
    cap = videoSourceInitialization()
    
    # Check video source is opened
    if not videoSourceSanityCheck(cap):
        print("Some required files could not be found for program...",
          "\nPlease contact with the manufacturer!")
        raise SystemExit
        
    main(cap)
    
