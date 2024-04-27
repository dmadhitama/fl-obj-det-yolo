def fix_bounding_box(bbox):
  """Fixes a bounding box list to ensure xmin <= xmax and ymin <= ymax.

  Args:
    bbox: A list of four floats representing [xmin, ymin, xmax, ymax].

  Returns:
    A corrected bounding box list.
  """
  xmin, ymin, xmax, ymax = bbox
  return [min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)]
