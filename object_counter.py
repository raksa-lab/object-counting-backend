import cv2
from ultralytics import YOLO

def object_counting(source_path, output_path=None, conf_threshold=0.5, iou_threshold=0.5):
    """
    Performs object detection and counting using YOLOv8 on an image or video.

    Args:
        source_path (str): Path to the input image or video file.
        output_path (str, optional): Path to save the output image/video. If None, displays the output.
        conf_threshold (float, optional): Confidence threshold for object detection. Defaults to 0.5.
        iou_threshold (float, optional): IoU threshold for Non-Maximum Suppression. Defaults to 0.5.
    """

    # Load a pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")  # You can choose other models like yolov8s.pt, yolov8m.pt, etc.

    # Open the video file or image
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"Error: Could not open source_path: {source_path}")
        return
    
    is_video = False
    if hasattr(cap, 'get') and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
        is_video = True

    if is_video:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object if output_path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    object_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame, conf=conf_threshold, iou=iou_threshold)

        # Process results
        current_frame_objects = {}
        for r in results:
            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                current_frame_objects[class_name] = current_frame_objects.get(class_name, 0) + 1

                # Update total counts (for video, this will be the count in the last frame)
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

            # Annotate the frame with bounding boxes and labels
            annotated_frame = r.plot()

        # Display counts on the frame
        y_offset = 30
        for obj, count in current_frame_objects.items():
            text = f"{obj}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30

        if output_path:
            if is_video:
                out.write(annotated_frame)
            else:
                cv2.imwrite(output_path, annotated_frame)
        else:
            # Note: cv2.imshow might not work in all environments (like headless servers)
            # cv2.imshow("Object Counting", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            pass

    # Release resources
    cap.release()
    if is_video and out:
        out.release()
    cv2.destroyAllWindows()

    print("\nFinal Object Counts:")
    for obj, count in object_counts.items():
        print(f"{obj}: {count}")

    return object_counts

if __name__ == "__main__":
    # Example usage:
    # For an image:
    # object_counting("path/to/your/image.jpg", output_path="output_image.jpg")

    # For a video:
    # object_counting("path/to/your/video.mp4", output_path="output_video.mp4")

    # To display output without saving (for images or videos):
    # object_counting("path/to/your/image.jpg")
    # object_counting("path/to/your/video.mp4")

    # Placeholder for user to provide input
    print("Please provide the path to your image or video file.")
    print("Example: python3 object_counter.py --source_path 'your_image.jpg' --output_path 'output.jpg'")
    print("Or: python3 object_counter.py --source_path 'your_video.mp4' --output_path 'output.mp4'")

    import argparse
    parser = argparse.ArgumentParser(description="Object Counting using YOLOv8.")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the input image or video file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output image/video. If None, displays the output.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")

    args = parser.parse_args()

    object_counting(args.source_path, args.output_path, args.conf_threshold, args.iou_threshold)
