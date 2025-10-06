#!/usr/bin/env python3
"""
Convert merged annotations JSON to CSV format for adding new features.
"""

import json
import csv
from pathlib import Path
from loguru import logger


def extract_damage_location(polygon_x, polygon_y):
    """Extract approximate damage location from polygon coordinates."""
    if not polygon_x or not polygon_y:
        return "unknown"

    min_x, max_x = min(polygon_x), max(polygon_x)
    min_y, max_y = min(polygon_y), max(polygon_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    img_width, img_height = 640, 480

    if center_x < img_width * 0.33:
        horizontal = "left"
    elif center_x > img_width * 0.67:
        horizontal = "right"
    else:
        horizontal = "center"

    if center_y < img_height * 0.33:
        vertical = "top"
    elif center_y > img_height * 0.67:
        vertical = "bottom"
    else:
        vertical = "middle"

    if horizontal == "center" and vertical == "middle":
        return "center"
    else:
        return f"{vertical}_{horizontal}"


def convert_json_to_csv(json_file, csv_file):
    """Convert VIA annotation JSON to CSV format."""
    logger.info(f"Loading annotations from: {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    logger.info(f"Processing {len(data)} images")

    csv_rows = []
    processed_count = 0

    for image_key, annotation in data.items():
        filename = annotation.get('filename', '')
        file_attributes = annotation.get('file_attributes', {})
        regions = annotation.get('regions', [])

        # Handle no_damage images
        if not regions and file_attributes.get('damage_class') == 'no_damage':
            csv_rows.append({
                'filename': filename,
                'damage_type': 'no_damage',
                'severity': 'none',
                'confidence': 1.0,
                'damage_location': 'full_image',
                'damage_severity': '',
                'damage_location_detailed': '',
                'damage_type_extended': '',
                'polygon_points_x': '',
                'polygon_points_y': '',
                'bbox_x1': '',
                'bbox_y1': '',
                'bbox_x2': '',
                'bbox_y2': ''
            })
            processed_count += 1
            continue

        # Process each damage region
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})

            polygon_x = shape_attrs.get('all_points_x', [])
            polygon_y = shape_attrs.get('all_points_y', [])

            if polygon_x and polygon_y:
                x1, y1 = min(polygon_x), min(polygon_y)
                x2, y2 = max(polygon_x), max(polygon_y)
            else:
                x1 = y1 = x2 = y2 = ''

            damage_type = region_attrs.get('damage_type', '')
            severity = region_attrs.get('severity', 'medium')
            confidence = region_attrs.get('confidence', 1.0)
            damage_location = extract_damage_location(polygon_x, polygon_y)

            row = {
                'filename': filename,
                'damage_type': damage_type,
                'severity': severity,
                'confidence': confidence,
                'damage_location': damage_location,
                'damage_severity': '',
                'damage_location_detailed': '',
                'damage_type_extended': '',
                'polygon_points_x': ','.join(map(str, polygon_x)) if polygon_x else '',
                'polygon_points_y': ','.join(map(str, polygon_y)) if polygon_y else '',
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2
            }

            csv_rows.append(row)
            processed_count += 1

        if processed_count % 1000 == 0:
            logger.info(f"Processed {processed_count} damage instances")

    logger.info(f"Total processed: {processed_count} damage instances")

    # Write to CSV
    if csv_rows:
        fieldnames = csv_rows[0].keys()

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    logger.info(f"CSV saved to: {csv_file}")
    logger.info(f"Total rows: {len(csv_rows)}")


def main():
    """Main function."""
    json_file = Path("data/processed/merged_annotations.json")
    csv_file = Path("data/processed/dataset_with_new_features.csv")

    if not json_file.exists():
        logger.error(f"Input file not found: {json_file}")
        return

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    convert_json_to_csv(json_file, csv_file)

    logger.info("Conversion completed successfully!")
    logger.info("You can now edit the CSV file to add values for the new features:")
    logger.info("- damage_severity: More detailed severity assessment")
    logger.info("- damage_location_detailed: More specific location information")
    logger.info("- damage_type_extended: Extended damage type classification")


if __name__ == "__main__":
    main()