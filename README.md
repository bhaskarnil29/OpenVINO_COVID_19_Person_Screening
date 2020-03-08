# How to run Python Program in Windows 10

 - Clone this Github reprository.
 - Open command prompt and initialize Openvino environment (run
   setupvars.bat).
 - Navigate to Github reprository folder in command prompt.
  - Run this command on command prompt ***python covid_19_person_screening.py --input demo_input.mp4  --thermal_camera demo_input_thermal.mp4  --cpu_ext cpu_extension_avx2.dll ***
 - ***--input path_to_input_video_file*** arg parameter is used to provide input video file from vision camera(i.e CCTV output of  airport) to perform inference on it.
 - ***--thermal_camera path_to_thermal_camera_file*** arg parameter is used to provide input video file from thermal camera, both thermal camera  and vision camera should have same view of field, resolution and FPS. If not , it need to pre-processed to match with o/p of vision camera.  Currently in program, thermal camera video is not from actual thermal camera and we are mimicking it by applying image processing on vision camera output.   
 - ***--cpu_ext path_to_cpu_extension_file*** arg parameter is used to provide cpu extension file. It is processor and OS dependent file so provide cpu extension file accordingly to your system configuration. 
 - Expected output will be:

   Two OpenCV video window will be opened, in first window vision camera video will be played along with face detection inference (face bounding box) and action recognition inference (with recognised action i.e. coughing). In second window thermal camera video will be played with face bounding box, bounding box coordinate take from vision camera frame. Once face bounding box is drawn on thermal frame, face temperature is read within bounding box. This face temperature is shown in vision frame. Also person is categorized as low risk(no fever) , moderate risk (fever but no coughing) and high risk(fever with coughing action). 
On command prompt recognised action from vision camera and face temperature from thermal camera is printed respectively. ** See  below snapshots **

![Inferred Frame of vision camera](https://github.com/chetancyber24/OpenVINO_COVID_19_Person_Screening/blob/master/output_screenshot/screenshot2.jpg)

 **Snapshot 1: Frame inferred in video where adult with fever is coughing and categorized as high risk **

![Frame of thermal camera](https://github.com/chetancyber24/OpenVINO_COVID_19_Person_Screening/blob/master/output_screenshot/screenshot3.jpg)

 **Snapshot 2: Thermal camera frame, based on face bounding box face temperature is read  **

![Command prompt output](https://github.com/chetancyber24/OpenVINO_COVID_19_Person_Screening/blob/master/output_screenshot/screenshot1.jpg)
 **Snapshot 3 : Python command prompt output showing thermal and vision camera frame's inferred output.**

 - Final inferred output video also will be saved in same input video directory suffixed with _inferred.avi

 - Above program is tested on Windows 10 environment. For Linux openvino
   setup  and cpu extension, please follow Openvino documentation.




# Demo Video
Python Program Demo Video can be accessed here : https://youtu.be/L89iBouHMLU

 


# Problem Statement

One among the biggest threats the world has been facing today is the virus Covid-19 which has already taken 3600 lives. Being a disease with no cure in the form of an antidote or cure medication according to World Health Organization (WHO), Covid-19 or CoronaVirus can only be prevented. This brings in the importance of having accurate screening systems in public areas such as airports ( especially as there are high chances of people coming from Corona affected countries), seaports and railway stations. 
Although WHO advises certain precautionary measures such as wearing an N95 mask, washing hands in frequent intervals, avoiding shake hands and physical contact with another person, we believe it is also equally important to let humans showing early symptoms of this virus to be sent to Quarantine for a minimum of 28 days.  
If we look at Wuhan city in China (where the Covid-19 is believed to have first started to spread) the city lockdown enabled to control the spread of the virus to a massive extend and new case reported daily for the virus has come to double digit compared to the three and four digits in late December and early January. 
With the virus spreading to other countries, installing accurate screening system in public place is gaining importance day by day in our fight with Covid-19
This is when we brainstormed the idea of identifying actions such as sneezing, coughing and blowing the nose of a human and understanding the body temperature as symptoms for initial screening.
To execute this we used thermal images + normal vision imaging camera to capture the actions of humans and detect symptoms like fever, coughing and sneezing using OpenVINO DNN model.




# Solution(Idea)

Our idea is to quickly screen Corona virus infected persons in public place(i.e. airport , mall etc) by using OpenVINO DNN models, vision camera and thermal camera. As corona virus symptoms are fever, coughing and sneezing . Our idea is to use thermal + normal vision imaging camera and OpenVINO DNN model  to detect these symptoms in person in public place to screen. We firstly use OpenVINO face-person-detection-retail model to detect individual human body bounding box and corresponding face bounding box in each frame of vision camera. Once we get bounding box for each face in vision frame, we look for corresponding face temperature in thermal camera using bounding box co-ordinate. Based on each person’s face temperature, we detect whether person has fever or not. If person has fever, we cropped his/her human body using human bounding box out of vision frame. These human body cropped images feed to OpenVINO action-recognition model to detect action of coughing, sneezing or blowing nose. (1) If coughing, sneezing or blowing nose action detected in person with high fever, person will be categorized as High Risk(Fever & Sneezing).(2)If coughing action not detected in person with fever, person will be categorized as moderate risk. ). (2) If coughing action not detected in person with fever, person will be categorized as moderate risk. (3) If person doesn’t have fever, person will be categorized as no risk.
 See below flow chart for pictorial representation, how this detection work.




# Flow Chart
![Algorithm Flow Chart](https://github.com/chetancyber24/OpenVINO_COVID_19_Person_Screening/blob/master/output_screenshot/flow_chart.jpg)







# Known Limitations &Issue

 -  Currently OpenVINO pre-trained action-recognition-0001 model is not able to detect coughing, sneezing action on input video. Possible reason could be that model’s training dataset(Kinetics-400 dataset) have very less videos of adult coughing and have more videos of child in dataset. Hence it is failing to detect adult’s coughing action in video. To overcome this issue in demo program, we are just hardcoding coughing action for demo purpose. For real deployment, action recognition model which can robustly detect adult’s coughing action. Also we are mimicking thermal camera output because we don't have actual thermal camera hardware.
  




# Project Members
Chetan Verma, Saikat Pandit & Lakshmi Prasannakumar 

# References:

 - Udacity Intel® Edge AI Scholarship Foundation Course
 - Intel OpenVino Documentation
 - Sample videos (shutterstock.com)




