import cv2,os,random
import math
import argparse
from agenet_helpers import load_to_IE, preprocessing
from agenet_inference import perform_inference

def get_face_temp(thermal_frame, personBox):
    #use box co-ordinate to detect face region in thermal camera frame
    #read temperature of center point in box 
    return 102 + round(random.random(),1) # Currently mimicking thermal temperature.

def get_action(frame):
    #Pass visual frame to OpenVino action recognition model to detect action like coughing, sneezing or blowing nose
    #Currently openvinoaction recognition model is not predicting right action
    action_dict = {0:'Coughing',1:'Sneezing',2:'Coughing',3:'Coughing',4:'Coughing',5:'Coughing',} #Mimicking action recognition model o/p
    return action_dict[random.randrange(0,5,1)]
def getpersons(frame,person_net , personnet_input_shape, conf_threshold =0.7):
    image=frame.copy()
    imageHeight=image.shape[0]
    imageWidth=image.shape[1]
    # cv2.imshow('Sample image', image)
    # cv2.waitKey(0)
    detected_persons =perform_inference(person_net, 's', image, personnet_input_shape)
    personBoxes=[]
    #print(detected_persons['detection_out'])
    #print(detected_persons['detection_out'].shape[2])
    
    for i in range(detected_persons['detection_out'].shape[2]):
        confidence =  detected_persons['detection_out'][0,0,i,2]
        
        if confidence>=conf_threshold:
            #print ('Confidence is',confidence)
            x1=int(detected_persons['detection_out'][0,0,i,3]*imageWidth)
            y1=int(detected_persons['detection_out'][0,0,i,4]*imageHeight)
            x2=int(detected_persons['detection_out'][0,0,i,5]*imageWidth)
            y2=int(detected_persons['detection_out'][0,0,i,6]*imageHeight)
            personBoxes.append([x1,y1,x2,y2])
            image=cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), int(round(imageHeight/150)), 8)
    return image , personBoxes
    



parser=argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--thermal_camera',default='demo_thermal_camera_out.avi')
curr_dir =os.getcwd()
cpu_ext_file_path = os.path.join(curr_dir,'.\cpu_extension_avx2.dll')
parser.add_argument('--cpu_ext',default=cpu_ext_file_path)
args=parser.parse_args()
CPU_EXTENSION = args.cpu_ext
#print('CPU EXtension',CPU_EXTENSION)

#openvino Face detection model loading
person_net , personnet_input_shape=load_to_IE('face-detection-retail-0004.xml', CPU_EXTENSION)



IS_IMAGE =False
if('.jpg' in args.input or '.bmp' in args.input or '.png' in args.input):
    IS_IMAGE = True 
    fps=0
    frame=cv2.imread(args.input)
    
else:
    video=cv2.VideoCapture(args.input if args.input else 0)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    file_path=args.input.split('.')
    thermal_video =cv2.VideoCapture(args.thermal_camera)
    del(file_path[len(file_path)-1])
    out_video_name = '.'.join(file_path)+'_inferred.avi'
    
    #print("FPS of video is ",fps)
    
    out_video=cv2.VideoWriter(out_video_name,cv2.CAP_OPENCV_MJPEG,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
padding=20
while cv2.waitKey(1)<0:
    if(not IS_IMAGE):
     hasFrame,frame=video.read()
     hasThermalFrame,thermal_frame=thermal_video.read()
     if not (hasFrame and hasThermalFrame):
         cv2.waitKey(1)
         break
    
    resultImg,personBoxes=getpersons(frame,person_net , personnet_input_shape)
    # print(type(frame))
    # cv2.imshow('Showing Frame',frame)
    # cv2.waitKey(0)
    
    for personBox in personBoxes:    
        
        
        x1=personBox[0]
        y1=personBox[1]
        x2=personBox[2]
        y2=personBox[3]
        face_temp = get_face_temp(thermal_frame, personBox)
        print ('Face temp in Fahrenheit is {}, Detected Fever'.format(face_temp))
        action = get_action(frame)
        print('Action Recognition : ' ,action) 
        cv2.rectangle(thermal_frame, (x1,y1), (x2,y2), (255,255,255), int(round(frame_height/150)), 10)
               
        cv2.putText(thermal_frame,'Reading Face temperature' , (personBox[0], personBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        temp_text = 'Face temp in Fahrenheit is ' + str(face_temp)
        action_text = 'Detected action is ' + str(action)
        
        cv2.putText(resultImg,temp_text , (20, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(resultImg,action_text , (20, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(resultImg,'High Risk (Fever & Coughing Detected)' , (personBox[0], personBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('COVID-19 Person Screnning, Inferred Output', resultImg)
        cv2.imshow('Thermal Camera O/P', thermal_frame)
    
    if(IS_IMAGE):
      cv2.waitKey(0)
      break
    else:
        None 
        out_video.write(resultImg)
    

if(not IS_IMAGE):
    video.release()
    out_video.release()
    thermal_video.release()
cv2.destroyAllWindows() 
