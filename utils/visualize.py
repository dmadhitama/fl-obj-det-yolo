import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_bboxes(image, bboxes, categories, color_list):
  """
  Visualizes bounding boxes on an image.

  Args:
    image: An image read by cv2.
    bboxes: List of bounding boxes in [xmin, ymin, xmax, ymax] format.
    categories: List of category names for each bounding box.
    color_list: (Dictionary) List of colors for each category.
  """
  # Load the image

  # Iterate over bounding boxes and categories
  for bbox, category in zip(bboxes, categories):
    xmin, ymin, xmax, ymax = map(int, bbox)
    # Draw the bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_list[category], 2)
    # Add the category label
    cv2.putText(image, category, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_list[category], 2)

  # Display the image
  plt.figure(figsize=(10, 10), dpi=200)
  plt.imshow(image)
  plt.axis("off")
  plt.show()
#   cv2.imshow("Image with Bounding Boxes", image)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
