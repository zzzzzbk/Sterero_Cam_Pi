import numpy as np
import cv2

# -------- USER SETTINGS --------
CALIB_NPZ = "stereo_calib_charuco.npz"
LEFT_IMG  = "calib\\left_09.png"
RIGHT_IMG = "calib\\right_09.png"

OUT_OVERLAY = "rectify_check_overlay.png"
OUT_RECTL   = "rectify_check_rectL.png"
OUT_RECTR   = "rectify_check_rectR.png"

DRAW_LINES_EVERY_PX = 40   # horizontal guide line spacing
SHOW_WINDOW = True         # set False if headless

# Optional: enable if your test images contain the checkerboard used for calibration
CHECK_CHESSBOARD = True
CHESSBOARD = (7, 5)        # inner corners (cols, rows) - must match your board
# --------------------------------
window_name = "Full2Kwindow"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 2304//2, 1296 //2)


def stack_with_lines(imgL, imgR, step=40):
    """Horizontally stack two images and draw horizontal lines across both."""
    if imgL.shape[:2] != imgR.shape[:2]:
        raise ValueError("Left/right rectified images have different sizes.")

    h, w = imgL.shape[:2]
    both = np.hstack([imgL, imgR])

    for y in range(0, h, step):
        cv2.line(both, (0, y), (2*w - 1, y), (0, 255, 0), 1)

    # vertical separator
    cv2.line(both, (w, 0), (w, h - 1), (255, 0, 0), 2)
    return both


def main():
    data = np.load(CALIB_NPZ, allow_pickle=True)

    image_size = tuple(data["image_size"])  # (w, h)
    mapLx, mapLy = data["mapLx"], data["mapLy"]
    mapRx, mapRy = data["mapRx"], data["mapRy"]

    imgL = cv2.imread(LEFT_IMG, cv2.IMREAD_COLOR)
    imgR = cv2.imread(RIGHT_IMG, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        raise RuntimeError("Failed to load left/right images (check filenames).")

    h, w = imgL.shape[:2]
    if (w, h) != image_size:
        raise RuntimeError(
            f"Image size {(w,h)} does not match calibration size {image_size}.\n"
            "Rectification maps are resolution-specific. Use the same capture resolution as calibration."
        )

    # Rectify
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    cv2.imwrite(OUT_RECTL, rectL)
    cv2.imwrite(OUT_RECTR, rectR)

    # Overlay epipolar lines
    overlay = stack_with_lines(rectL, rectR, step=DRAW_LINES_EVERY_PX)
    cv2.imwrite(OUT_OVERLAY, overlay)
    print("Saved:")
    print(" ", OUT_RECTL)
    print(" ", OUT_RECTR)
    print(" ", OUT_OVERLAY)

    if SHOW_WINDOW:
        cv2.imshow(window_name, overlay)
        print("Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optional quantitative check using chessboard corners
    if CHECK_CHESSBOARD:
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        okL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD, flags)
        okR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD, flags)

        if not (okL and okR):
            print("[Chessboard check] Chessboard not found in both rectified images.")
            return

        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), crit)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), crit)

        dy = (cornersL[:, 0, 1] - cornersR[:, 0, 1])
        print("[Chessboard check] Vertical mismatch Δy (pixels):")
        print("  mean:", float(np.mean(dy)))
        print("  std :", float(np.std(dy)))
        print("  max abs:", float(np.max(np.abs(dy))))

        # Visualize detected corners
        visL = rectL.copy()
        visR = rectR.copy()
        cv2.drawChessboardCorners(visL, CHESSBOARD, cornersL, True)
        cv2.drawChessboardCorners(visR, CHESSBOARD, cornersR, True)
        both2 = np.hstack([visL, visR])
        cv2.imwrite("rectify_check_chessboard.png", both2)
        print("Saved: rectify_check_chessboard.png")


if __name__ == "__main__":
    main()
