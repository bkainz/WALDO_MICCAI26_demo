"""
Visualization utilities for WALDO.

Provides functions for drawing bounding boxes, creating comparison figures,
and generating publication-quality visualizations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import io
import base64


class BoundingBoxVisualizer:
    """Draw bounding boxes on images with customizable styles."""

    def __init__(
        self,
        gt_color: str = "green",
        pred_color: str = "red",
        line_width: int = 3,
        show_labels: bool = True,
        font_size: int = 12
    ):
        """
        Initialize visualizer.

        Args:
            gt_color: Color for ground truth boxes
            pred_color: Color for predicted boxes
            line_width: Width of box outlines
            show_labels: Whether to show labels on boxes
            font_size: Font size for labels
        """
        self.gt_color = gt_color
        self.pred_color = pred_color
        self.line_width = line_width
        self.show_labels = show_labels
        self.font_size = font_size

        # Try to load a nice font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            self.font = ImageFont.load_default()

    def draw_boxes(
        self,
        image: np.ndarray,
        gt_boxes: Optional[List[List[float]]] = None,
        pred_boxes: Optional[List[List[float]]] = None,
        iou_score: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (H, W, 3)
            gt_boxes: Ground truth boxes [[x1, y1, x2, y2], ...]
            pred_boxes: Predicted boxes [[x1, y1, x2, y2], ...]
            iou_score: Optional IoU score to display

        Returns:
            Image with boxes drawn as numpy array
        """
        # Convert to PIL
        if image.max() <= 1.0:
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(image.astype(np.uint8))

        draw = ImageDraw.Draw(img_pil)

        # Draw GT boxes
        if gt_boxes:
            for i, box in enumerate(gt_boxes):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=self.gt_color, width=self.line_width)
                if self.show_labels:
                    label = f"GT {i+1}"
                    draw.text((x1, y1-15), label, fill=self.gt_color, font=self.font)

        # Draw predicted boxes (dashed effect by drawing multiple offset lines)
        if pred_boxes:
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = box
                # Solid outline
                draw.rectangle([x1, y1, x2, y2], outline=self.pred_color, width=self.line_width)
                if self.show_labels:
                    label = f"Pred {i+1}"
                    draw.text((x1, y2+5), label, fill=self.pred_color, font=self.font)

        # Draw IoU score if provided
        if iou_score is not None:
            iou_text = f"IoU: {iou_score:.3f}"
            # Draw on top-right corner with background
            text_bbox = draw.textbbox((0, 0), iou_text, font=self.font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = img_pil.width - text_width - 10
            y = 10

            # Background rectangle
            draw.rectangle(
                [x-5, y-5, x+text_width+5, y+text_height+5],
                fill='black',
                outline='white',
                width=2
            )
            draw.text((x, y), iou_text, fill='white', font=self.font)

        return np.array(img_pil)

    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        gt_boxes_list: List[List[List[float]]],
        pred_boxes_list: List[List[List[float]]],
        iou_scores: List[float],
        grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create a grid comparison of multiple results.

        Args:
            images: List of images
            titles: List of titles for each image
            gt_boxes_list: List of GT boxes for each image
            pred_boxes_list: List of predicted boxes for each image
            iou_scores: List of IoU scores
            grid_size: (rows, cols) or None for auto

        Returns:
            Grid image as numpy array
        """
        n = len(images)

        # Auto-calculate grid size
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size

        # Draw boxes on each image
        annotated_images = []
        for i in range(n):
            img_with_boxes = self.draw_boxes(
                images[i],
                gt_boxes_list[i],
                pred_boxes_list[i],
                iou_scores[i]
            )
            annotated_images.append(img_with_boxes)

        # Create grid
        h, w = annotated_images[0].shape[:2]
        grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255

        for idx, img in enumerate(annotated_images):
            row = idx // cols
            col = idx % cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

        # Add titles
        grid_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)

        for idx, title in enumerate(titles):
            if idx >= n:
                break
            row = idx // cols
            col = idx % cols
            x = col * w + 10
            y = row * h + h - 30
            draw.text((x, y), title, fill='blue', font=self.font)

        return np.array(grid_pil)


class InteractiveVisualizer:
    """Create interactive HTML visualizations for Jupyter notebooks."""

    @staticmethod
    def create_html_comparison(
        image: np.ndarray,
        gt_boxes: List[List[float]],
        pred_boxes: List[List[float]],
        iou: float,
        image_id: str
    ) -> str:
        """
        Create HTML visualization with hoverable boxes.

        Args:
            image: Image array
            gt_boxes: Ground truth boxes
            pred_boxes: Predicted boxes
            iou: IoU score
            image_id: Image identifier

        Returns:
            HTML string
        """
        # Convert image to base64
        if image.max() <= 1.0:
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(image.astype(np.uint8))

        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        h, w = image.shape[:2]

        # Build SVG overlay
        svg_elements = []

        # GT boxes
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            svg_elements.append(f'''
                <rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"
                      fill="none" stroke="green" stroke-width="3"
                      class="gt-box" data-box-id="gt-{i}">
                    <title>GT Box {i+1}</title>
                </rect>
            ''')

        # Predicted boxes
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box
            svg_elements.append(f'''
                <rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"
                      fill="none" stroke="red" stroke-width="2" stroke-dasharray="5,5"
                      class="pred-box" data-box-id="pred-{i}">
                    <title>Predicted Box {i+1}</title>
                </rect>
            ''')

        svg_overlay = '\n'.join(svg_elements)

        # HTML template
        html = f'''
        <div class="waldo-viz" style="position: relative; display: inline-block;">
            <img src="data:image/png;base64,{img_base64}"
                 style="max-width: 100%; height: auto;">
            <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
                {svg_overlay}
            </svg>
            <div style="margin-top: 10px;">
                <strong>Image:</strong> {image_id}<br>
                <strong>IoU:</strong> {iou:.3f}<br>
                <strong>GT Boxes:</strong> {len(gt_boxes)} |
                <strong>Predicted:</strong> {len(pred_boxes)}
            </div>
        </div>
        <style>
            .gt-box:hover {{ stroke-width: 5; }}
            .pred-box:hover {{ stroke-width: 4; }}
        </style>
        '''

        return html


def save_results_figure(
    results: List[Dict],
    output_path: str,
    n_samples: int = 9,
    metric: str = "iou"
):
    """
    Save a figure showing best and worst results.

    Args:
        results: List of result dictionaries with 'image', 'gt_boxes', 'pred_boxes', 'iou'
        output_path: Where to save the figure
        n_samples: Number of samples to show (best and worst)
        metric: Metric to sort by
    """
    # Sort by metric
    sorted_results = sorted(results, key=lambda x: x.get(metric, 0), reverse=True)

    # Get best and worst
    n_best = n_samples // 2
    n_worst = n_samples - n_best
    selected = sorted_results[:n_best] + sorted_results[-n_worst:]

    # Create visualization
    visualizer = BoundingBoxVisualizer()

    images = [r['image'] for r in selected]
    titles = [f"{r['image_id']}: {metric}={r[metric]:.3f}" for r in selected]
    gt_boxes = [r['gt_boxes'] for r in selected]
    pred_boxes = [r['pred_boxes'] for r in selected]
    iou_scores = [r['iou'] for r in selected]

    grid = visualizer.create_comparison_grid(
        images, titles, gt_boxes, pred_boxes, iou_scores
    )

    # Save
    Image.fromarray(grid).save(output_path)
    print(f"Saved results figure to {output_path}")


def create_method_overview_figure(
    query_image: np.ndarray,
    reference_images: List[np.ndarray],
    predicted_boxes: List[List[float]],
    output_path: str
):
    """
    Create a figure showing the WALDO method overview.

    Args:
        query_image: Query image with anomaly
        reference_images: Selected healthy references
        predicted_boxes: Predicted anomaly boxes
        output_path: Where to save figure
    """
    # Create layout: Query | Refs | Result
    n_refs = len(reference_images)
    h, w = query_image.shape[:2]

    # Create canvas
    canvas_w = w * (2 + n_refs)
    canvas = np.ones((h, canvas_w, 3), dtype=np.uint8) * 255

    # Place query
    canvas[:, :w] = (query_image * 255).astype(np.uint8)

    # Place references
    for i, ref in enumerate(reference_images):
        x_start = w * (i + 1)
        canvas[:, x_start:x_start+w] = (ref * 255).astype(np.uint8)

    # Place result (query with boxes)
    visualizer = BoundingBoxVisualizer(show_labels=False)
    result = visualizer.draw_boxes(query_image, pred_boxes=predicted_boxes)
    x_start = w * (n_refs + 1)
    canvas[:, x_start:x_start+w] = result

    # Add labels
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "Query", fill='white', font=font)
    for i in range(n_refs):
        draw.text((w*(i+1) + 10, 10), f"Ref {i+1}", fill='white', font=font)
    draw.text((w*(n_refs+1) + 10, 10), "Result", fill='white', font=font)

    # Save
    canvas_pil.save(output_path)
    print(f"Saved method overview to {output_path}")
