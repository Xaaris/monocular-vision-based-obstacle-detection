from dataclasses import dataclass


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def get_width(self) -> int:
        return self.x2 - self.x1

    def get_height(self) -> int:
        return self.y2 - self.y1

    def get_center(self) -> tuple:
        return int(self.x1 + self.get_width() / 2), int(self.y1 + self.get_height() / 2)

    def get_position_in_image(self) -> tuple:
        from Main import INPUT_IMAGE_DIMENSIONS
        return self.get_center()[0] / INPUT_IMAGE_DIMENSIONS[0], self.get_center()[1] / INPUT_IMAGE_DIMENSIONS[1]
