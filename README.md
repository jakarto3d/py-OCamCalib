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
* **Calibration parameters**: Calibration parameters are saved in,json file to better portability.
* **Camera model**: Once calibration is done, the camera class is ready to use. Load the calibration file and use all the predefined mapping 
 functions ( world to pixel, pixel to world, undistorted, equirectangular projection, ...) in your project.

## installation

```commandline
git clone https://github.com/jakarto3d/py-OCamCalib.git
cd py-OCamCalib

# for conda user 
conda env create --file environment.yml
conda activate RoadSignsDetection

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
python main.py ./test_images/fish_1 8 6  --camera-name fisheye_1
```

### Use case 2 : Automatic detection and check 
```commandline
python main.py ./test_images/fish_1 8 6  --camera-name fisheye_1 --check
```

### Use case 3 : Load corners from file  
```commandline
python main.py ./test_images/fish_1 8 6  --camera-name fisheye_1 --corners-path ./py_ocamcalib/checkpoints/corners_detection/detections_fisheye_1_18052022_153543.pickle
```