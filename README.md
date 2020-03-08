# How to run Python Program in Windows 10

 - Clone this Github reprository.
 - Open command prompt and initialize Openvino environment (run
   setupvars.bat).
 - Navigate to Github reprository folder in command prompt.
  - Run this command on command prompt: **python covid_19_person_screening.py --input demo_input.mp4  --thermal_camera demo_input_thermal.avi  --cpu_ext cpu_extension_avx2.dll **
 - ***--input path_to_input_video_file*** arg parameter is used to provide input video file from vision camera(i.e CCTV output of  airport) to perform inference on it.
-***--thermal_camera path_to_thermal_camera_file*** arg parameter is used to provide input video file from thermal camera, both thermal camera  and vision camera should have same view of field, resolution and FPS. If not , it need to pre-processed to match with o/p of vision camera.  Currently in program, thermal camera video is not from actual thermal camera and we are mimicking it by applying image processing on vision camera output.   
 - ***--cpu_ext path_to_cpu_extension_file*** arg parameter is used to provide cpu extension file. It is processor and OS dependent file so provide cpu extension file accordingly to your system configuration. 
 - Expected output will be:

   Two OpenCV video window will be opened, in first window vision camera video will be played along with face detection inference (face bounding box) and action recognition inference (with recognised action i.e. coughing). In second window thermal camera video will be played with face bounding box, bounding box coordinate take from vision camera frame. Once face bounding box is drawn on thermal frame, face temperature is read within bounding box. This face temperature is shown in vision frame. Also person is categorized as low risk(no fever) , moderate risk (fever but no coughing) and high risk(fever with coughing action). 
On command prompt recognised action from vision camera and face temperature from thermal camera is printed respectively. ** See  below snapshot **

	Final inferred output video also will be saved in same input video directory suffixed with _inferred.avi

 - Above program is tested on Windows 10 environment. For Linux openvino
   setup  and cpu extension, please follow Openvino documentation.



# Output Snapshots

![Command prompt output](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot1.jpg)
 **Snapshot 1 : Python command prompt output showing each frame inferred status(Adult present or not).** 

![Inferred Frame with adult](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot2.jpg)
**Snapshot 2: Frame inferred in video where adult is present in car with kid.**

![Inferred Frame with alone kid](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot3.jpg)
 **Snapshot 3: Frame inferred in video where alone kid is present in car with warning embedded.**


# Demo Video
Python Program Demo Video can be accessed here : https://www.youtube.com/watch?v=978tXgmopO4&feature=youtu.be

 


# Problem Statement

In recent time Corona virus has caused significant deaths and distress.  One of challenge is to screen human infected with Corona virus symptoms in public place (i.e. airport, railway station, shopping malls etc) so than infected person can be pathologically tested and if found positive person need to be quarantine and treated as early as possible. Quarantine is very important step to stop spread of corona virus.



# Solution(Idea)

Our idea is to screen Corona virus infected persons in public place(i.e. airport , mall etc) to use vision camera along with thermal camera. As corona virus symptoms are fever, coughing and sneezing . Our idea is to use thermal + normal vision imaging camera and OpenVINO DNN model  to detect these symptoms in person in public place to screen. We firstly use OpenVINO face-person-detection-retail model to detect individual human body bounding box and corresponding face bounding box in each frame of vision camera. Once we get bounding box for each face in vision frame, we look for corresponding face temperature in thermal camera using bounding box co-ordinate. Based on each person’s face temperature, we detect whether person has fever or not. If person has fever, we cropped his/her human body using human bounding box out of vision frame. These human body cropped images feed to OpenVINO action-recognition model to detect action of coughing, sneezing or blowing nose. (1) If coughing, sneezing or blowing nose action detected in person with high fever, person will be categorized as High Risk(Fever & Sneezing).(2)If coughing action not detected in person with fever, person will be categorized as moderate risk. ). (2) If coughing action not detected in person with fever, person will be categorized as moderate risk. (3) If person doesn’t have fever, person will be categorized as no risk.
 See below flow chart for pictorial representation, how this detection work.


For face detect model, Openvino pretrained model **face-detection-retail-0004** is used and for age prediction we used **[Gil Levi and Tal Hassner Age Classification](https://talhassner.github.io/home/publication/2015_CVPR)** Using Convolutional Neural Network. Age prediction model is available as Caffe model, we converted it to Openvino IR format using model optimizer.

# Flow Chart
![Algorithm Flow Chart](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/flow_chart.png)







# Known Issue, Possible Solutions & Further enhancement

 - In some random frame of video age prediction model make wrong
   prediction and classify adult into kid (or vice versa).  To supress
   false alarm, kid alone alarm will be raised after certain amount of
   time where consecutive frame is detected kid with no adult present.
   In python program, this threshold parameter is determined by
   ALONE_KID_TIME_THREESHOLD it is set to 2sec in program because of
   short demo video. In real life this can be set to slightly longer
   time (i.e. 15 mins).
  
 - Currently when kid is detected without adult, as alarm video is   
   embedded with warning text. In real life this should trigger warning 
   sms or app notification to parents’ phone or 911 emergency responder 
   using V2X technology (connected vehicle).
 - This further can be enhanced to disable specific airbag when small
   kid is detected.
 - Camera mounted on rear view mirror capturing view of inside car cabin
   will not be able to capture child sat in **rear facing car seat**. To
   solve this issue one more camera(red camera in below snapshot), need
   to mount on near back windshield to capture kid sitting in rear
   facing seat.
   ![Camera in back windshield](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot4.jpg)
**Figure: Camera mounted on near back windshield to capture kid seating in rear facing car seat.**


# Google Colab Project(Alternative)
As a different approach we also tried installing OpenVino in Google Colab and using Res10 Single Shot Detector Caffe Model for face detection and the same Age Classification Network.

Steps to run the project in Google Colab:
 - Access the Folder for the project from the following link:
Child_Lost_In_Car
 - Open the colab notebook: ‘ Detecting_Kids_Left_Alone_In_a_Car.ipynb’
 - Change the RunType to GPU
Menubar → RunTime → Change runtime type
 - Execute each cell by cell ( cntrl + enter) or (alt+enter) or `'/>'` button near each cells
Or Menubar → RunTime → run all
   
 - Please Note: After running or the execution of the first cell for
   mounting the Google drive in  either step 3 or step 4 you choose, you will be prompted to click on the link generated to   authenticate the mounting of the Google Drive
 - Read through the Colab Notebook to download the output file and see the output.

# Project Members
Chetan Verma, Lakshmi Prasannakumar & Saikat Pandit

# References:

 - Udacity Intel® Edge AI Scholarship Foundation Course
 - Intel OpenVino Documentation
 - Age detection Model
   ([https://data-flair.training/blogs/python-project-gender-age-detection/](https://data-flair.training/blogs/python-project-gender-age-detection/))
 - Sample videos (shutterstock.com)




