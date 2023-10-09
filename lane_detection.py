import cv2
import numpy as np


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=(255, 255, 255), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def main():
    # Load the video
    cap = cv2.VideoCapture(
        "D:\lane detection\mixkit-going-down-a-curved-highway-through-a-mountain-range-41576-medium.mp4"
    )

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Define region of interest (ROI)
        height, width = edges.shape
        roi_vertices = [
            (0, height),
            (width / 2, height / 2),
            (width, height),
        ]
        roi = region_of_interest(edges, np.array([roi_vertices], np.int32))

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=50,
        )

        # Create a blank image to draw the detected lines on
        line_img = np.zeros_like(frame)

        # Draw the detected lines on the blank image
        if lines is not None:
            draw_lines(line_img, lines)

        # Overlay the detected lines on the original frame
        result = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)

        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
