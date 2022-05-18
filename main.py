from py_ocamcalib.calibration import CalibrationEngine
import typer


def main(working_dir: str,
         chessboard_size_row: int,
         chessboard_size_column: int,
         corners_path: str = None,
         check: bool = False,
         camera_name: str = "MyCamera",
         square_size: float = 1):

    """
    :param chessboard_size_row: Number of INNER corners per a chessboard on a row \n
    :param chessboard_size_column: Number of INNER corners per a chessboard on a column \n
    :param working_dir: Path to the folder which contains all the chessboard pattern of your camera.\n
    :param corners_path: If None perform detection. Otherwise, detection is skipped and calibration is done with corners
    saved inside file at corners_path.
    :param check: If True, manual corners annotations can be performed in addition to automatic detection to prevent
    potential miss detection.\n
    :param camera_name: name of your camera.\n
    :param square_size: size of a chessboard square side. If you don't know this size, calibration will be done in
    chessboard square unit. Otherwise, if chessboard square side is 30 mm for example, calibration will be done in mm.
    :return:
    """

    chessboard_size = (chessboard_size_row, chessboard_size_column)
    my_calib_engine = CalibrationEngine(working_dir, chessboard_size, camera_name, square_size)

    if corners_path is None:
        my_calib_engine.detect_corners(check=check)
        my_calib_engine.save_detection()
    else:
        my_calib_engine.load_detection(corners_path)

    my_calib_engine.estimate_fisheye_parameters()
    my_calib_engine.find_poly_inv()
    my_calib_engine.save_calibration()
    my_calib_engine.show_model_projection()
    my_calib_engine.show_reprojection_error()
    my_calib_engine.show_reprojection()


if __name__ == "__main__":
    typer.run(main)
