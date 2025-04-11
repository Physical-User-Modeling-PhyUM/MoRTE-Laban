import csv
import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kinect joint order (25 joints)
KINECT_JOINT_ORDER = [
    'SPINE_BASE',  # 0 - Calculated as midpoint between hips
    'SPINE_MID',  # 1 - Midpoint between SPINE_BASE and SPINE_SHOULDER
    'NECK',  # 2 - Midpoint between shoulders (approximation)
    'HEAD',  # 3 - MediaPipe's HEAD landmark
    'LEFT_SHOULDER',  # 4 - MediaPipe's LEFT_SHOULDER
    'LEFT_ELBOW',  # 5 - MediaPipe's LEFT_ELBOW
    'LEFT_WRIST',  # 6 - MediaPipe's LEFT_WRIST
    'LEFT_HAND',  # 7 - MediaPipe's LEFT_PINKY (approximation)
    'RIGHT_SHOULDER',  # 8 - MediaPipe's RIGHT_SHOULDER
    'RIGHT_ELBOW',  # 9 - MediaPipe's RIGHT_ELBOW
    'RIGHT_WRIST',  # 10 - MediaPipe's RIGHT_WRIST
    'RIGHT_HAND',  # 11 - MediaPipe's RIGHT_PINKY (approximation)
    'LEFT_HIP',  # 12 - MediaPipe's LEFT_HIP
    'LEFT_KNEE',  # 13 - MediaPipe's LEFT_KNEE
    'LEFT_ANKLE',  # 14 - MediaPipe's LEFT_ANKLE
    'LEFT_FOOT',  # 15 - MediaPipe's LEFT_FOOT_INDEX
    'RIGHT_HIP',  # 16 - MediaPipe's RIGHT_HIP
    'RIGHT_KNEE',  # 17 - MediaPipe's RIGHT_KNEE
    'RIGHT_ANKLE',  # 18 - MediaPipe's RIGHT_ANKLE
    'RIGHT_FOOT',  # 19 - MediaPipe's RIGHT_FOOT_INDEX
    'SPINE_SHOULDER',  # 20 - MediaPipe's midpoint between shoulders
    'LEFT_HAND_TIP',  # 21 - MediaPipe's LEFT_INDEX
    'LEFT_THUMB',  # 22 - MediaPipe's LEFT_THUMB
    'RIGHT_HAND_TIP',  # 23 - MediaPipe's RIGHT_INDEX
    'RIGHT_THUMB'  # 24 - MediaPipe's RIGHT_THUMB
]


def calculate_midpoint(landmark1, landmark2, image_width=1, image_height=1):
    """Calculate midpoint between two landmarks"""
    return [
        (landmark1.x * image_width + landmark2.x * image_width) / 2,
        (landmark1.y * image_height + landmark2.y * image_height) / 2,
        (landmark1.z + landmark2.z) / 2  # z remains normalized
    ]


def infer_spine_base(landmarks, image_width=1, image_height=1):
    """Infer SPINE_BASE as midpoint between hips"""
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    return calculate_midpoint(left_hip, right_hip, image_width, image_height)


def infer_spine_mid(landmarks, spine_base, image_width=1, image_height=1):
    """Infer SPINE_MID as midpoint between SPINE_BASE and SPINE_SHOULDER"""
    spine_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    spine_shoulder2 = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    spine_shoulder_mid = calculate_midpoint(spine_shoulder, spine_shoulder2, image_width, image_height)

    # Calculate midpoint
    return [
        (spine_base[0] + spine_shoulder_mid[0]) / 2,
        (spine_base[1] + spine_shoulder_mid[1]) / 2,
        (spine_base[2] + spine_shoulder_mid[2]) / 2
    ]


def infer_neck(landmarks, image_width=1, image_height=1):
    """Infer NECK as midpoint between shoulders (approximation) at height of mouth"""
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    neck = calculate_midpoint(left_shoulder, right_shoulder, image_width, image_height)

    mouth_left = landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
    mouth_right = landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
    neck[1] = calculate_midpoint(mouth_left, mouth_right, image_width, image_height)[1]

    return neck

def infer_head(landmarks, image_width=1, image_height=1):
    """Infer HEAD as midpoint between eyes (approximation) at depth of mid shoulders"""
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    neck = calculate_midpoint(left_shoulder, right_shoulder, image_width, image_height)

    left_eye = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    head = calculate_midpoint(left_eye, right_eye, image_width, image_height)
    head[0] = neck[0]
    head[2] = neck[2]

    return head


def get_landmark_visibility(landmark):
    """Get visibility of a landmark"""
    return landmark.visibility if hasattr(landmark, 'visibility') else 0.5


def convert_landmarks_to_kinect_format(landmarks, timestamp, image_width=1, image_height=1, use_pixel_coords=False):
    """
    Convert MediaPipe landmarks to Kinect format with inferred joints.

    Args:
        landmarks: MediaPipe pose landmarks
        timestamp: Timestamp for the frame
        image_width: Width of the image in pixels (required if use_pixel_coords=True)
        image_height: Height of the image in pixels (required if use_pixel_coords=True)
        use_pixel_coords: If True, output coordinates will be in pixel space

    Returns:
        List of values in Kinect CSV format
    """
    # Initialize all joints with default values (0, 0, 0, 0)
    joint_data = {joint: [0.0, 0.0, 0.0, 0] for joint in KINECT_JOINT_ORDER}

    if landmarks:
        # First calculate inferred joints that other joints depend on
        spine_base = infer_spine_base(landmarks, image_width, image_height)
        spine_mid = infer_spine_mid(landmarks, spine_base, image_width, image_height)
        neck = infer_neck(landmarks, image_width, image_height)
        head = infer_head(landmarks, image_width, image_height)

        # Map all joints including inferred ones
        joint_mappings = {
            'SPINE_BASE': spine_base,
            'SPINE_MID': spine_mid,
            'NECK': neck,
            'HEAD': head,
            'LEFT_SHOULDER': landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
            'LEFT_ELBOW': landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
            'LEFT_WRIST': landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
            'LEFT_HAND': landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY],
            'RIGHT_SHOULDER': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'RIGHT_ELBOW': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
            'RIGHT_WRIST': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
            'RIGHT_HAND': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY],
            'LEFT_HIP': landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
            'LEFT_KNEE': landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
            'LEFT_ANKLE': landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
            'LEFT_FOOT': landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
            'RIGHT_HIP': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
            'RIGHT_KNEE': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
            'RIGHT_ANKLE': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
            'RIGHT_FOOT': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
            'SPINE_SHOULDER': calculate_midpoint(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                image_width, image_height
            ),
            'LEFT_HAND_TIP': landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX],
            'LEFT_THUMB': landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB],
            'RIGHT_HAND_TIP': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX],
            'RIGHT_THUMB': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
        }

        for joint, mp_data in joint_mappings.items():
            if isinstance(mp_data, list):  # For inferred points that are already coordinates
                x, y, z = mp_data
                visibility = 0.7  # Medium confidence for inferred points
            else:  # For direct MediaPipe landmarks
                if use_pixel_coords:
                    x = mp_data.x * image_width
                    y = mp_data.y * image_height
                else:
                    x = mp_data.x
                    y = mp_data.y
                z = mp_data.z if hasattr(mp_data, 'z') else 0
                visibility = get_landmark_visibility(mp_data)

            new_x = -z
            new_y = x
            new_z = y
            x = new_x
            y = new_y
            z = new_z
            if not use_pixel_coords:
                # Convert coordinates to Kinect-like space (only if using normalized coords)
                # MediaPipe coordinates are normalized (0-1), we'll convert to meters
                # Assuming a rough estimate that 1.0 in normalized coordinates
                z = (z - 0.5) * 1  # Convert to meters, centered around origin
                y = (0.5 - y) * 1  # Flip Y axis to match Kinect's coordinate system
                x = x * 1  # Convert to meters

            # Determine tracking state based on visibility
            if visibility > 0.7:
                tracking_state = 2  # Tracked
            elif visibility > 0.3:
                tracking_state = 1  # Inferred
            else:
                tracking_state = 0  # Not tracked

            joint_data[joint] = [x, y, z, tracking_state]

    # Apply Z-axis rotation if requested
    joint_data = rotate_points(joint_data, 90, 'x')
    joint_data = rotate_points(joint_data, 30, 'z')

    # Flatten the data in the correct Kinect joint order
    flattened_data = [timestamp]
    for joint in KINECT_JOINT_ORDER:
        flattened_data.extend(joint_data[joint])

    return flattened_data


def rotate_points(points, angle_degrees, axis='z'):
    """
    Rotate points around a specified axis by a given angle.

    Args:
        points: Dictionary of joint data {joint_name: [x, y, z, tracking_state]}
        angle_degrees: Rotation angle in degrees (positive is counter-clockwise)
        axis: Rotation axis ('x', 'y', or 'z')

    Returns:
        Rotated points dictionary
    """
    angle_rad = np.radians(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotated_points = {}

    for joint, data in points.items():
        x, y, z, tracking_state = data

        if axis.lower() == 'z':
            new_x = x * cos_theta - y * sin_theta
            new_y = x * sin_theta + y * cos_theta
            new_z = z
        elif axis.lower() == 'x':
            new_y = y * cos_theta - z * sin_theta
            new_z = y * sin_theta + z * cos_theta
            new_x = x
        elif axis.lower() == 'y':
            new_x = x * cos_theta + z * sin_theta
            new_z = -x * sin_theta + z * cos_theta
            new_y = y
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'")

        rotated_points[joint] = [new_x, new_y, new_z, tracking_state]

    return rotated_points

def process_video_to_kinect_csv(video_path, output_csv, use_pixel_coords=False):
    """
    Process a video file and save pose data in Kinect CSV format.

    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
        use_pixel_coords: If True, output coordinates will be in pixel space
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties for pixel conversion if needed
    if use_pixel_coords:
        image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        image_width = 1
        image_height = 1

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        # header = ['Timestamp']
        # for joint in KINECT_JOINT_ORDER:
        #     header.extend([f'{joint}_X', f'{joint}_Y', f'{joint}_Z', f'{joint}_TrackingState'])
        # csv_writer.writerow(header)

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (MediaPipe uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = pose.process(frame_rgb)

            # Create timestamp (in hundred-nanosecond units)
            timestamp = int((time.time() - start_time) * 1e7)  # Convert seconds to 100ns units

            # Convert to Kinect format
            kinect_data = convert_landmarks_to_kinect_format(
                results.pose_landmarks,
                timestamp,
                image_width,
                image_height,
                use_pixel_coords
            )

            # Write to CSV
            csv_writer.writerow(kinect_data)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f'Processed frame {frame_count}')

    cap.release()
    print(f'Processing complete. Data saved to {output_csv}')


if __name__ == '__main__':
    input_video = 'input_video.webm'
    output_csv = 'kinect_pose_data.csv'

    process_video_to_kinect_csv(input_video, output_csv, use_pixel_coords=False)