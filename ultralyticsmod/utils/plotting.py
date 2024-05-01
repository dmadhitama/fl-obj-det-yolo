from ultralytics.utils.plotting import Annotator
import torch
import cv2

class AnnotatorMeasurement(Annotator):
    def get_center(
            self,
            box,
        ):
        """Extracts the center coordinates
        
        Args:
            box: bounding box coordinates (x1, y1, x2, y2)

        Returns:
            Extracted center coordinates (x, y)
        """
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        # estimate box center point
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return center_x, center_y
    
    def estimate_coordinate(
            self,
            centers, 
            center_ref, 
            units, 
        ):
        """Estimate coordinates of bounding box based on units

        Args:
            centers: center coordinates of bounding box (x, y)
            center_ref: center coordinate of reference (x, y)
            units: (x, y) unit (coef) scalar of coordinates

        Returns:
            New coordinates of bounding box based on units (x, y)
        """
        # return coordinates is based on displayed axis on image
        return (
            float(center_ref[0] - centers[0])/float(units[0]), 
            float(centers[1] - center_ref[1])/float(units[1]), 
        )
    
    def estimate_box_size(
            self,
            box, # (x1, y1, x2, y2) coordinates of bounding box
            units, # (x, y) unit (coef) scalar of coordinates
        ):
        """Estimate the size of a bounding box
        
        Args:
            box: coordinate of bounding box (x1, y1, x2, y2)
            units: (x, y) unit (coef) scalar of coordinates

        Returns:
            Estimated box size (x, y)
        """
        return (
            float(box[2] - box[0])/float(units[0]),
            float(box[3] - box[1])/float(units[1]),
        )
    
    def draw_coordinate_label(
            self,
            img, 
            coordinate, 
            scaled_coordinate, 
            text, 
            font, 
            font_scale,
            thickness,
            color,
    ):
        """Draw coordinates label
        
        Args:
            img: image to be labeled
            coordinate: coordinate (x, y)
            scaled_coordinate: scaled coordinate to be labeled (x, y)
            text: text to write on the label
            font: font to use
            font_scale: font scale
            thickness: thickness of the label
            color: color of the label

        Returns:
            Image with the label drawn
        """
        text_size, _ = cv2.getTextSize(
            text, 
            font, 
            font_scale, 
            thickness
        )
        text_w, text_h = text_size
        img = cv2.rectangle(
            img,
            (int(coordinate[0]),int(coordinate[1])-10),
            (int(coordinate[0]) + int(text_w), int(coordinate[1]) + int(text_h)),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.putText(
            img,
            f'[{round(scaled_coordinate[0],2)}, {round(scaled_coordinate[1],2)}]', 
            (int(coordinate[0]),int(coordinate[1])+5),
            fontFace=font, 
            fontScale=font_scale,
            color=(255,255,255),
            thickness=thickness,
        )
        return img