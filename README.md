<p align="center">
  <img src="./docs/logo.png" alt="Typer">
</p>

Py-OCamCalib is a pure Python/Numpy implementation of <a href="https://rpg.ifi.uzh.ch/people_scaramuzza.html">Scaramuzzas</a> 
<a href="https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab">OcamCalib</a> Toolbox.

📚 This work is based on: \
Scaramuzza, D., Martinelli, A. and Siegwart, R., (2006). <a href="https://rpg.ifi.uzh.ch/docs/ICVS06_scaramuzza.pdf">"A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion", Proceedings of IEEE International Conference of Vision Systems (ICVS'06), New York, January 5-7, 2006. </a>\
Urban, S.; Leitloff, J.; Hinz, S. (2015): <a href="https://www.ipf.kit.edu/downloads/Preprint_ImprovedWideAngleFisheyeAndOmnidirectionalCameraCalibration.pdf">Improved Wide-Angle, Fisheye and Omnidirectional Camera Calibration. ISPRS Journal of Photogrammetry and Remote Sensing 108, 72-79.</a>

The key features are:

* **Easy to use**: It's easy to use for the final users. Two lines in the terminal.
* **Chessboard corners detection**: Automatic chessboard corners detection and optional manual correction to prevent miss detection.
* **Calibration parameters**: Calibration parameters are saved in .json file to better portability.
* **Camera model**: Once calibration is done, the camera class is ready to use. Load the calibration file and use all the predefined mapping 
 functions (world to pixel, pixel to world, undistorted, equirectangular projection, ...) in your project.

## Installation

```commandline
git clone https://github.com/jakarto3d/py-OCamCalib.git
cd py-OCamCalib

# for conda user 
conda env create --file environment.yml
conda activate py-OCamCalib

# for virtualenv user 
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Example
./test_images contains images of chessboard pattern taken from three differences fisheye lens.
You can use it to test the project.

### Use case 1 : Automatic detection and no check 
```commandline
python calibration_script.py ./../../test_images/fish_1 8 6  --camera-name fisheye_1
```

### Use case 2 : Automatic detection and check 
```commandline
python calibration_script.py ./../../test_images/fish_1 8 6  --camera-name fisheye_1 --check
```
⚠️**Manual corners verification is not verify intuitive !**

*Instructions*:

Once the opencv windows is opened, you can enter in two different mode : SELECTION MODE and DRAW MODE.
 
 **SELECTION MODE**: Press 's' to enter the mode. Selection mode allows you to select points that have not been detected
 accurately. After pressing 's' you can surround such point with bounding box by pressing left mouse, drawing
 bounding boxes **AND** confirm by pressing enter. Once you select all your points, you have to quit selection mode by
 pressing 'esc'. Selected points should appear RED. 

 **DRAW MODE**: Press 'd' each time you want draw the new point and click to drop off your modified points. New point 
 have to be drawn in same order then selected points.

 When work is done, press 'z' to quit. A windows with your modified pattern should appear.

### Use case 3 : Load corners from file  
```commandline
python calibration_script.py ./../../test_images/fish_1 8 6  --camera-name fisheye_1 --corners-path ./../checkpoints/corners_detection/detections_fisheye_1_09092022_053310.pickle
```