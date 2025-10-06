#!/usr/bin/env python3
"""
COCO Annotation Tool with GUI
Converts YOLO format to COCO and allows annotation of:
- damage_location (string)
- damage_severity (int)
- damage_type (string)
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import yaml


class COCOAnnotationTool:
    """GUI tool for annotating images with COCO format + custom attributes."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("COCO Annotation Tool - Collision Parts Dataset")
        self.root.geometry("1400x900")
        
        # Data storage
        self.images_dir = None
        self.labels_dir = None
        self.output_file = None
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.current_photo = None
        self.class_names = []
        
        # COCO dataset structure
        self.coco_data = {
            "info": {
                "description": "Collision Parts Dataset with Damage Annotations",
                "version": "1.0",
                "year": 2025,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.annotation_id = 1
        self.image_id_map = {}
        
        # Predefined options
        self.damage_locations = [
            "front_bumper", "rear_bumper", "hood", "trunk",
            "door_front_left", "door_front_right", "door_rear_left", "door_rear_right",
            "fender_front_left", "fender_front_right", "fender_rear_left", "fender_rear_right",
            "windshield", "rear_window", "headlight", "taillight",
            "mirror", "wheel", "roof", "side_panel", "other"
        ]
        
        self.damage_types = [
            "scratch", "dent", "crack", "broken", "shattered",
            "crushed", "paint_damage", "rust", "missing", "deformed", "other"
        ]
        
        # Current annotations for displayed image
        self.current_annotations = []
        self.selected_annotation_index = None
        
        # Setup GUI
        self.setup_ui()
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Control-s>', lambda e: self.save_annotations())
        self.root.bind('<Delete>', lambda e: self.delete_current_annotation())
        
    def setup_ui(self):
        """Setup the GUI layout."""
        
        # Top menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Images Directory", command=self.load_images_directory)
        file_menu.add_command(label="Load Class Names", command=self.load_class_names)
        file_menu.add_command(label="Save Annotations (Ctrl+S)", command=self.save_annotations)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image display
        left_frame = ttk.Frame(main_frame, width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image canvas
        self.canvas = tk.Canvas(left_frame, bg='gray', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Navigation controls
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="← Previous (Left Arrow)", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        self.image_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_label.pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="Next (Right Arrow) →", command=self.next_image).pack(side=tk.RIGHT, padx=5)
        
        # Right panel - Annotation controls
        right_frame = ttk.Frame(main_frame, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Title
        ttk.Label(right_frame, text="Annotation Panel", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Current annotations list
        list_frame = ttk.LabelFrame(right_frame, text="Detected Objects", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollable listbox
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.annotations_listbox = tk.Listbox(list_frame, yscrollcommand=list_scroll.set, height=10)
        self.annotations_listbox.pack(fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.annotations_listbox.yview)
        self.annotations_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)
        
        # Annotation form
        form_frame = ttk.LabelFrame(right_frame, text="Annotation Details", padding=10)
        form_frame.pack(fill=tk.BOTH, pady=(0, 10))
        
        # Damage Location
        ttk.Label(form_frame, text="Damage Location:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.location_var = tk.StringVar()
        self.location_combo = ttk.Combobox(form_frame, textvariable=self.location_var, 
                                           values=self.damage_locations, state='readonly', width=25)
        self.location_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Damage Severity
        ttk.Label(form_frame, text="Damage Severity (0-10):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.severity_var = tk.IntVar(value=5)
        severity_frame = ttk.Frame(form_frame)
        severity_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.severity_scale = ttk.Scale(severity_frame, from_=0, to=10, variable=self.severity_var, 
                                        orient=tk.HORIZONTAL, length=150)
        self.severity_scale.pack(side=tk.LEFT)
        self.severity_label = ttk.Label(severity_frame, text="5", width=3)
        self.severity_label.pack(side=tk.LEFT, padx=5)
        self.severity_scale.config(command=lambda v: self.severity_label.config(text=str(int(float(v)))))
        
        # Damage Type
        ttk.Label(form_frame, text="Damage Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.damage_type_var = tk.StringVar()
        self.damage_type_combo = ttk.Combobox(form_frame, textvariable=self.damage_type_var,
                                              values=self.damage_types, state='readonly', width=25)
        self.damage_type_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Update button
        ttk.Button(form_frame, text="Update Annotation", command=self.update_annotation).grid(
            row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Delete (Del)", command=self.delete_current_annotation).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(action_frame, text="Save All (Ctrl+S)", command=self.save_annotations).pack(
            side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Load images directory to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="No data", justify=tk.LEFT)
        self.stats_label.pack()
        
    def load_class_names(self):
        """Load class names from YAML file."""
        filename = filedialog.askopenfilename(
            title="Select Class Names File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = yaml.safe_load(f)
                    self.class_names = data.get('names', [])
                
                # Initialize COCO categories
                self.coco_data["categories"] = []
                for i, name in enumerate(self.class_names):
                    self.coco_data["categories"].append({
                        "id": i,
                        "name": name,
                        "supercategory": "vehicle_part"
                    })
                
                self.status_var.set(f"Loaded {len(self.class_names)} classes from {Path(filename).name}")
                messagebox.showinfo("Success", f"Loaded {len(self.class_names)} class names")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load class names: {e}")
    
    def load_images_directory(self):
        """Load images directory and corresponding YOLO labels."""
        images_dir = filedialog.askdirectory(title="Select Images Directory")
        
        if not images_dir:
            return
        
        self.images_dir = Path(images_dir)
        
        # Try to find labels directory
        possible_labels = [
            self.images_dir.parent / "labels",
            self.images_dir.with_name("labels"),
            self.images_dir / "labels"
        ]
        
        self.labels_dir = None
        for labels_path in possible_labels:
            if labels_path.exists():
                self.labels_dir = labels_path
                break
        
        if not self.labels_dir:
            # Ask user
            labels_dir = filedialog.askdirectory(title="Select Labels Directory (YOLO format)")
            if labels_dir:
                self.labels_dir = Path(labels_dir)
        
        # Load image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(self.images_dir.glob(f"*{ext}"))
            self.image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        self.image_files = sorted(self.image_files)
        
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the selected directory")
            return
        
        # Ask for output file
        self.output_file = filedialog.asksaveasfilename(
            title="Save COCO Annotations As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="annotations.json"
        )
        
        if not self.output_file:
            self.output_file = str(self.images_dir.parent / "annotations.json")
        
        # Try to load existing annotations if file exists
        if Path(self.output_file).exists():
            try:
                with open(self.output_file, 'r') as f:
                    existing_data = json.load(f)
                    if messagebox.askyesno("Load Existing", "Found existing annotations. Load them?"):
                        self.coco_data = existing_data
                        self.annotation_id = max([a['id'] for a in self.coco_data['annotations']], default=0) + 1
                        
                        # Build image ID map
                        for img in self.coco_data['images']:
                            self.image_id_map[img['file_name']] = img['id']
            except Exception as e:
                print(f"Could not load existing annotations: {e}")
        
        # Load first image
        self.current_index = 0
        self.load_current_image()
        
        self.status_var.set(f"Loaded {len(self.image_files)} images from {self.images_dir.name}")
        self.update_stats()
        
    def load_current_image(self):
        """Load and display the current image with annotations."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_index]
        self.image_label.config(text=f"Image {self.current_index + 1} / {len(self.image_files)}: {image_path.name}")
        
        # Load image
        self.current_image = Image.open(image_path)
        img_width, img_height = self.current_image.size
        
        # Add to COCO images if not already present
        if image_path.name not in self.image_id_map:
            image_id = len(self.coco_data['images']) + 1
            self.image_id_map[image_path.name] = image_id
            
            self.coco_data['images'].append({
                "id": image_id,
                "file_name": image_path.name,
                "width": img_width,
                "height": img_height,
                "date_captured": datetime.now().isoformat()
            })
        else:
            image_id = self.image_id_map[image_path.name]
        
        # Load YOLO annotations if available
        label_path = self.labels_dir / f"{image_path.stem}.txt" if self.labels_dir else None
        
        # Get existing COCO annotations for this image
        existing_annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        
        # If no COCO annotations but YOLO labels exist, convert them
        if not existing_annotations and label_path and label_path.exists():
            self.current_annotations = self.load_yolo_annotations(label_path, img_width, img_height, image_id)
        else:
            self.current_annotations = existing_annotations
        
        # Draw image with bounding boxes
        self.draw_image()
        
        # Update annotations list
        self.update_annotations_list()
        
        # Update stats
        self.update_stats()
        
    def load_yolo_annotations(self, label_path: Path, img_width: int, img_height: int, image_id: int) -> List[Dict]:
        """Convert YOLO annotations to COCO format."""
        annotations = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = [float(x) for x in parts[1:5]]
                    
                    # Convert to pixel coordinates
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    # Convert to COCO bbox format [x_min, y_min, width, height]
                    x_min = x_center_px - width_px / 2
                    y_min = y_center_px - height_px / 2
                    
                    # Create COCO annotation
                    annotation = {
                        "id": self.annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, width_px, height_px],
                        "area": width_px * height_px,
                        "iscrowd": 0,
                        "segmentation": [],
                        # Custom attributes
                        "damage_location": "",
                        "damage_severity": 0,
                        "damage_type": ""
                    }
                    
                    annotations.append(annotation)
                    self.annotation_id += 1
        
        except Exception as e:
            print(f"Error loading YOLO annotations: {e}")
        
        return annotations
    
    def draw_image(self):
        """Draw the image with bounding boxes on canvas."""
        if not self.current_image:
            return
        
        # Create a copy for drawing
        img_copy = self.current_image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Draw bounding boxes
        for i, ann in enumerate(self.current_annotations):
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            
            # Color based on whether it's annotated
            has_annotations = (ann.get('damage_location') and 
                             ann.get('damage_severity') is not None and 
                             ann.get('damage_type'))
            
            color = 'green' if has_annotations else 'red'
            if i == self.selected_annotation_index:
                color = 'yellow'
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
            
            # Draw label
            class_name = self.class_names[ann['category_id']] if ann['category_id'] < len(self.class_names) else f"Class {ann['category_id']}"
            draw.text((x_min, y_min - 15), f"{i+1}. {class_name}", fill=color)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.current_photo = ImageTk.PhotoImage(img_copy)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.current_photo)
        
    def update_annotations_list(self):
        """Update the listbox with current annotations."""
        self.annotations_listbox.delete(0, tk.END)
        
        for i, ann in enumerate(self.current_annotations):
            class_name = self.class_names[ann['category_id']] if ann['category_id'] < len(self.class_names) else f"Class {ann['category_id']}"
            
            location = ann.get('damage_location', '')
            severity = ann.get('damage_severity', 0)
            damage_type = ann.get('damage_type', '')
            
            status = "✓" if (location and severity > 0 and damage_type) else "○"
            
            list_text = f"{status} {i+1}. {class_name} | {location or '?'} | Sev:{severity} | {damage_type or '?'}"
            self.annotations_listbox.insert(tk.END, list_text)
    
    def on_annotation_select(self, event):
        """Handle annotation selection from listbox."""
        selection = self.annotations_listbox.curselection()
        if not selection:
            return
        
        self.selected_annotation_index = selection[0]
        ann = self.current_annotations[self.selected_annotation_index]
        
        # Populate form
        self.location_var.set(ann.get('damage_location', ''))
        self.severity_var.set(ann.get('damage_severity', 0))
        self.damage_type_var.set(ann.get('damage_type', ''))
        
        # Redraw image to highlight selected
        self.draw_image()
    
    def update_annotation(self):
        """Update the selected annotation with form values."""
        if self.selected_annotation_index is None:
            messagebox.showwarning("No Selection", "Please select an annotation from the list")
            return
        
        ann = self.current_annotations[self.selected_annotation_index]
        
        # Update values
        ann['damage_location'] = self.location_var.get()
        ann['damage_severity'] = self.severity_var.get()
        ann['damage_type'] = self.damage_type_var.get()
        
        # Update in main COCO data
        for i, coco_ann in enumerate(self.coco_data['annotations']):
            if coco_ann['id'] == ann['id']:
                self.coco_data['annotations'][i] = ann
                break
        else:
            # Not in COCO data yet, add it
            self.coco_data['annotations'].append(ann)
        
        # Refresh display
        self.update_annotations_list()
        self.draw_image()
        self.update_stats()
        
        self.status_var.set(f"Updated annotation #{self.selected_annotation_index + 1}")
        
        # Auto-select next annotation
        if self.selected_annotation_index < len(self.current_annotations) - 1:
            self.annotations_listbox.selection_clear(0, tk.END)
            self.annotations_listbox.selection_set(self.selected_annotation_index + 1)
            self.annotations_listbox.activate(self.selected_annotation_index + 1)
            self.on_annotation_select(None)
    
    def delete_current_annotation(self):
        """Delete the selected annotation."""
        if self.selected_annotation_index is None:
            return
        
        ann = self.current_annotations[self.selected_annotation_index]
        
        # Remove from current annotations
        del self.current_annotations[self.selected_annotation_index]
        
        # Remove from COCO data
        self.coco_data['annotations'] = [a for a in self.coco_data['annotations'] if a['id'] != ann['id']]
        
        # Clear selection
        self.selected_annotation_index = None
        
        # Refresh display
        self.update_annotations_list()
        self.draw_image()
        self.update_stats()
        
        self.status_var.set("Deleted annotation")
    
    def save_annotations(self):
        """Save all annotations to COCO JSON file."""
        if not self.output_file:
            messagebox.showerror("Error", "No output file specified")
            return
        
        # Update current image annotations in COCO data
        if self.current_annotations and self.image_files:
            image_path = self.image_files[self.current_index]
            image_id = self.image_id_map.get(image_path.name)
            
            # Remove old annotations for this image
            self.coco_data['annotations'] = [
                a for a in self.coco_data['annotations'] 
                if a['image_id'] != image_id
            ]
            
            # Add current annotations
            self.coco_data['annotations'].extend(self.current_annotations)
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.coco_data, f, indent=2, ensure_ascii=False)
            
            self.status_var.set(f"Saved annotations to {Path(self.output_file).name}")
            messagebox.showinfo("Success", f"Annotations saved to:\n{self.output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")
    
    def next_image(self):
        """Go to next image."""
        if not self.image_files:
            return
        
        # Save current annotations first
        if self.current_annotations and self.image_files:
            image_path = self.image_files[self.current_index]
            image_id = self.image_id_map.get(image_path.name)
            
            # Update COCO data
            self.coco_data['annotations'] = [
                a for a in self.coco_data['annotations'] 
                if a['image_id'] != image_id
            ]
            self.coco_data['annotations'].extend(self.current_annotations)
        
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.selected_annotation_index = None
        self.load_current_image()
    
    def prev_image(self):
        """Go to previous image."""
        if not self.image_files:
            return
        
        # Save current annotations first
        if self.current_annotations and self.image_files:
            image_path = self.image_files[self.current_index]
            image_id = self.image_id_map.get(image_path.name)
            
            # Update COCO data
            self.coco_data['annotations'] = [
                a for a in self.coco_data['annotations'] 
                if a['image_id'] != image_id
            ]
            self.coco_data['annotations'].extend(self.current_annotations)
        
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.selected_annotation_index = None
        self.load_current_image()
    
    def update_stats(self):
        """Update statistics display."""
        total_images = len(self.image_files) if self.image_files else 0
        total_annotations = len(self.coco_data['annotations'])
        
        # Count fully annotated
        fully_annotated = sum(
            1 for ann in self.coco_data['annotations']
            if ann.get('damage_location') and ann.get('damage_severity') > 0 and ann.get('damage_type')
        )
        
        stats_text = f"""Total Images: {total_images}
Total Annotations: {total_annotations}
Fully Annotated: {fully_annotated} / {total_annotations}
Progress: {(fully_annotated / total_annotations * 100) if total_annotations > 0 else 0:.1f}%"""
        
        self.stats_label.config(text=stats_text)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = COCOAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
