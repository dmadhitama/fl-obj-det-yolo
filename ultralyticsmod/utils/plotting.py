from ultralytics.utils.plotting import Annotator
import torch

class AnnotatorMeasurement(Annotator):
    def check(self):
        print("Checking mode")
        pass

    def get_center(
            self,
            box,
            label="",
        ):
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        # estimate box center point
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return center_x, center_y
    
    def estimate_coordinate(
            self,
            centers, # (x, y) center coordinates of bounding box
            center_ref, # (x, y) center coordinate of reference
            units, # (x, y) unit (coef) scalar of coordinates
        ):
        # return coordinates is based on displayed axis on image
        return (
            float(center_ref[0] - centers[0])/float(units[0]), 
            float(centers[1] - center_ref[1])/float(units[1]), 
        )