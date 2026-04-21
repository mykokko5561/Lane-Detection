# --- SELECTİNG DOTS WİTH MOUSE ---
points = []
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # Left clicked
        points.append((x, y))
        print(f"Nokta Seçildi: {x, y}")

# It takes a screenshot of the first second of the video.
success, img = vidcap.read()
cv2.namedWindow("Nokta Sec ve Kapat")
cv2.setMouseCallback("Nokta Sec ve Kapat", select_points)

while len(points) < 4: # Wait until 4 points are selected.
    cv2.imshow("Nokta Sec ve Kapat", img)
    cv2.waitKey(1)

cv2.destroyWindow("Nokta Sec ve Kapat")
# Assign the selected points as tl, bl, tr, br in an order
tl, bl, tr, br = points[0], points[1], points[2], points[3]
