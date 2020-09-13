"""Miscellaneous utility functions for exporting data"""

import csv
from typing import Optional

from model.DetectedObjects import DetectedObjects
from model.ObjectInstance import ObjectInstance


def write_detected_objects_to_csv(detected_objects: DetectedObjects, filename):
    file = open(filename + ".csv", 'w')

    with file:
        writer = csv.writer(file)

        for key, ot in detected_objects.objects.items():
            row = [key, ot.class_name, ot.active]
            occurrences: [Optional[ObjectInstance]] = ot.occurrences
            for occ in occurrences:
                row += ["None"] if occ is None else [occ.roi.get_center()]
            writer.writerow(row)
