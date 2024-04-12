import cv2


def draw_lines(image, pts):
    pts = pts.reshape((-1, 1, 2))

    isClosed = True

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv2.polylines(image, [pts],
                          isClosed, color, thickness)

    return image


def draw_img(image, list_points):
    for pts in list_points:
        image = draw_lines(image, pts)

    return image
