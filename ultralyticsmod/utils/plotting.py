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
    
    def label_center(
            self,
            box,
            label="",
        ):
        center_x, center_y = self.get_center(box)