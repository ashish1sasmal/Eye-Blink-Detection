# Eye-Blink-Detection
<b> “ The time in your life can go by in the blink of an eye, so look for what lasts a lifetime.  “

-- J.R. Rim

</b>

### The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

  - Facial landmarks
  
   - The right eye using [36, 42].
  
   - The left eye with [42, 48].

<img src="https://github.com/ashish1sasmal/Eye-Blink-Detection/blob/master/face_landmarks.jpg" width=320>

#### The Formula to detect blinking : 
  <img src="https://github.com/ashish1sasmal/Eye-Blink-Detection/blob/master/blink_detection_equation.png" width=320>
  
  ##### Conditions :
      If EAR <= 0.3:
           Blink Occur
 

<img src="https://github.com/ashish1sasmal/Eye-Blink-Detection/blob/master/Results/eye_blink_live.gif" width=500>
