"""Miscellaneous utility functions for exporting data"""

import csv
import math
from typing import Optional

from model.DetectedObjects import DetectedObjects
from model.ObjectInstance import ObjectInstance


def write_detected_objects_to_csv(detected_objects: DetectedObjects, prefix=""):
    for key, ot in detected_objects.objects.items():
        file = open("export/csv_export/" + prefix + "_" + ot.class_name + "_" + str(key) + ".csv", 'w')
        with file:
            writer = csv.writer(file)
            header = ["id", "class_name", "present", "confidence", "speed (m/s)", "velocity", "roi_center", "distance",
                      "3d_position", "# keypoints", "# descriptors"]
            writer.writerow(header)
            occurrences: [Optional[ObjectInstance]] = ot.occurrences
            velocity_rounded = "-" if not ot.is_present() else tuple(map(lambda e: round(e, 2), ot.get_velocity()))
            speed = "-" if not ot.is_present() else round(math.sqrt(sum([e ** 2 for e in velocity_rounded])), 2)

            for occ in occurrences:
                row = [key,
                       ot.class_name,
                       ot.is_present(),
                       "-" if occ is None else occ.confidence_score,
                       speed,
                       velocity_rounded,
                       "-" if occ is None else [occ.roi.get_center()],
                       "-" if occ is None else occ.approximate_distance(),
                       "-" if occ is None else tuple(map(lambda e: round(e, 2), occ.get_3d_position())),
                       "-" if occ is None or occ.keypoints is None else len(occ.keypoints),
                       "-" if occ is None or occ.descriptors is None else len(occ.descriptors)
                       ]
                writer.writerow(row)
