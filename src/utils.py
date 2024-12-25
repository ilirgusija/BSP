import math
import numpy as np
import matplotlib.pyplot as plt

@staticmethod
def convert_to_rectangle(l, width, min_rand, max_rand):
    map_width = max_rand - min_rand  # Assuming square map

    # Calculate the top, bottom, left, and right boundaries of the rectangle
    top = max_rand - l
    bottom = l
    left = (map_width - width) / 2
    right = left + width

    # Check for valid rectangle within map bounds
    if bottom < min_rand or top > max_rand or width > map_width:
        raise ValueError(
            "Invalid rectangle dimensions: exceeds map bounds.")

    return left, right, bottom, top

@staticmethod
def plot_rectangle(rectangle, color="-b"):
    left, right, bottom, top = rectangle

    # Plot the rectangle
    plt.fill([left, right, right, left, left], [bottom, bottom, top, top, bottom], color)

@staticmethod
def calc_distance_and_angle(parent, child):
    dx = child[0] - parent[0]
    dy = child[1] - parent[1]
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    return length, angle

@staticmethod
def generate_rectangle_from_reference(reference, length, orientation="horizontal", translation=(10, 0)):
    x_ref, y_ref = reference
    dx, dy = translation
    x_translated = x_ref + dx
    y_translated = y_ref + dy

    if orientation == "horizontal":
        # Fixed height of 10, horizontal length is variable
        left = x_translated-length/2
        right = x_translated + length/2
        bottom = y_translated - 2.5
        top = y_translated + 2.5
    elif orientation == "vertical":
        # Fixed width of 10, vertical length is variable
        left = x_translated - 2.5
        right = x_translated + 2.5
        bottom = y_translated-length/2
        top = y_translated + length/2
    else:
        raise ValueError(
            "Invalid orientation. Choose 'horizontal' or 'vertical'.")

    return left, right, bottom, top
