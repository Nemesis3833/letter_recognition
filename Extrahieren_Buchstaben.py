import cv2
import numpy as np
import os

output_folder = "extracted_images"
image = cv2.imread("a/arda_korkmaz-bilder-1 - Copy.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

horizontal_lines = []
vertical_lines = []

for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y1 - y2) < abs(x1 - x2):
            horizontal_lines.append((x1, y1, x2, y2))
        else:
            vertical_lines.append((x1, y1, x2, y2))

horizontal_lines.sort(key=lambda line: line[1])
vertical_lines.sort(key=lambda line: line[0])

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'
]

for row in range(len(horizontal_lines) - 1):
    for col in range(len(vertical_lines) - 1):
        x_start = vertical_lines[col][0]
        y_start = horizontal_lines[row][1]
        x_end = vertical_lines[col + 1][0]
        y_end = horizontal_lines[row + 1][1]

        letter_image = gray[y_start:y_end, x_start:x_end]

        letter = letters[row]
        filename = f"{output_folder}/Arda{letter}{col}.png"
        cv2.imwrite(filename, letter_image)

cv2.imshow('Cropped Image', letter_image)
cv2.waitKey(0)
cv2.destroyAllWindows()