import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

class ObjectTracker:
    """Track objects across multiple frames to build historical trajectories."""
    
    def __init__(self, max_distance=2.0, history_length=4):
        """
        Args:
            max_distance: maximum distance (meters) to match objects between frames
            history_length: number of historical positions to keep (for 2s at 0.5s intervals = 4)
        """
        self.max_distance = max_distance
        self.history_length = history_length
        self.tracks = {}  # track_id -> list of (timestamp, x, y, vx, vy, detection_name)
        self.next_id = 1
    
    def update(self, detections, timestamp, ego_pose=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: list of detection dicts
            timestamp: current timestamp
            ego_pose: ego vehicle pose for coordinate transformation
        
        Returns:
            list of tracked objects with IDs and history
        """
        # Transform detections to ego frame
        current_objects = []
        for det in detections:
            if ego_pose:
                rel_pos = self.transform_global_to_ego(det['translation'], ego_pose)
                x, y = rel_pos
            else:
                x, y = det['translation'][0], det['translation'][1]
            
            vx, vy = det.get('velocity', [0.0, 0.0])[:2]
            
            current_objects.append({
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'detection_name': det['detection_name'],
                'score': det.get('detection_score', 0.0),
                'original': det
            })
        
        # Match current detections with existing tracks
        tracked_objects = self._match_and_update(current_objects, timestamp)
        
        return tracked_objects
    
    def _match_and_update(self, current_objects, timestamp):
        """Match current detections with existing tracks using Hungarian algorithm."""
        if not current_objects:
            return []
        
        # Get active tracks (those seen recently)
        active_tracks = {tid: hist for tid, hist in self.tracks.items() 
                        if len(hist) > 0}
        
        tracked_objects = []
        
        if not active_tracks:
            # No existing tracks, create new ones
            for obj in current_objects:
                if obj['score'] >= 0.3:  # Filter by score
                    track_id = self.next_id
                    self.next_id += 1
                    self.tracks[track_id] = [(timestamp, obj['x'], obj['y'], obj['vx'], obj['vy'], obj['detection_name'])]
                    tracked_objects.append({
                        'id': track_id,
                        'x': obj['x'],
                        'y': obj['y'],
                        'vx': obj['vx'],
                        'vy': obj['vy'],
                        'detection_name': obj['detection_name'],
                        'history': []
                    })
        else:
            # Match using distance
            track_ids = list(active_tracks.keys())
            track_positions = np.array([[hist[-1][1], hist[-1][2]] for hist in active_tracks.values()])
            current_positions = np.array([[obj['x'], obj['y']] for obj in current_objects])
            
            # Compute distance matrix
            distances = cdist(track_positions, current_positions)
            
            # Simple greedy matching (for better results, use Hungarian algorithm)
            matched_tracks = set()
            matched_objects = set()
            
            for _ in range(min(len(track_ids), len(current_objects))):
                min_dist = distances.min()
                if min_dist > self.max_distance:
                    break
                
                track_idx, obj_idx = np.unravel_index(distances.argmin(), distances.shape)
                
                track_id = track_ids[track_idx]
                obj = current_objects[obj_idx]
                
                # Update track
                self.tracks[track_id].append((timestamp, obj['x'], obj['y'], obj['vx'], obj['vy'], obj['detection_name']))
                if len(self.tracks[track_id]) > self.history_length + 1:
                    self.tracks[track_id].pop(0)
                
                # Build history (excluding current position)
                history = [(h[1], h[2]) for h in self.tracks[track_id][:-1]]
                
                tracked_objects.append({
                    'id': track_id,
                    'x': obj['x'],
                    'y': obj['y'],
                    'vx': obj['vx'],
                    'vy': obj['vy'],
                    'detection_name': obj['detection_name'],
                    'history': history
                })
                
                matched_tracks.add(track_idx)
                matched_objects.add(obj_idx)
                
                # Mark as matched
                distances[track_idx, :] = np.inf
                distances[:, obj_idx] = np.inf
            
            # Create new tracks for unmatched objects
            for obj_idx, obj in enumerate(current_objects):
                if obj_idx not in matched_objects and obj['score'] >= 0.3:
                    track_id = self.next_id
                    self.next_id += 1
                    self.tracks[track_id] = [(timestamp, obj['x'], obj['y'], obj['vx'], obj['vy'], obj['detection_name'])]
                    tracked_objects.append({
                        'id': track_id,
                        'x': obj['x'],
                        'y': obj['y'],
                        'vx': obj['vx'],
                        'vy': obj['vy'],
                        'detection_name': obj['detection_name'],
                        'history': []
                    })
        
        return tracked_objects
    
    def transform_global_to_ego(self, translation, ego_pose):
        """Transform global coordinates to ego-vehicle coordinates."""
        ego_x, ego_y, ego_z = ego_pose['translation']
        quat = ego_pose['rotation']
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        dx = translation[0] - ego_x
        dy = translation[1] - ego_y
        
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        
        ego_x_rel = dx * cos_yaw - dy * sin_yaw
        ego_y_rel = dx * sin_yaw + dy * cos_yaw
        
        return [ego_x_rel, ego_y_rel]


def format_tracked_objects_to_nl(tracked_objects, time_step=0.5):
    """
    Convert tracked objects to natural language description.
    
    Args:
        tracked_objects: list of tracked object dicts with ID and history
        time_step: time interval for velocity (default 0.5s)
    
    Returns:
        formatted string description
    """
    if not tracked_objects:
        return "No notable objects detected."
    
    # Sort by distance
    for obj in tracked_objects:
        obj['distance'] = np.sqrt(obj['x']**2 + obj['y']**2)
    
    tracked_objects.sort(key=lambda x: x['distance'])
    
    # Build natural language description
    lines = ["The current percepted notable objects are listed here: Percepted Notable objects:"]
    
    for obj in tracked_objects:
        obj_id = obj['id']
        detection_name = obj['detection_name']
        x = obj['x']
        y = obj['y']
        vx = obj['vx'] * time_step
        vy = obj['vy'] * time_step
        
        # Format the line
        line = f"- {detection_name} (ID {obj_id}) at ({x:.1f}, {y:.1f}), velocity ({vx:.1f}, {vy:.1f}) per {time_step}s"
        
        # Add historical trajectory if available
        if obj['history']:
            history_str = ", ".join([f"({hx:.1f}, {hy:.1f})" for hx, hy in obj['history']])
            line += f"; historical trajectory: past {len(obj['history']) * time_step:.1f}s: {history_str}"
        
        lines.append(line)
    
    lines.append("Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. unit is meter")
    
    return "\n".join(lines)


def process_sequential_results(results_dict, timestamps_dict, ego_poses_dict=None, output_file=None):
    """
    Process results sequentially to build trajectories.
    
    Args:
        results_dict: dict mapping sample_token to list of detections
        timestamps_dict: dict mapping sample_token to timestamp
        ego_poses_dict: dict mapping sample_token to ego pose (optional)
        output_file: path to save output (optional)
    
    Returns:
        dict mapping sample_token to natural language description
    """
    # Sort samples by timestamp
    sorted_samples = sorted(timestamps_dict.items(), key=lambda x: x[1])
    
    tracker = ObjectTracker()
    nl_descriptions = {}
    
    for sample_token, timestamp in sorted_samples:
        if sample_token not in results_dict:
            continue
        
        detections = results_dict[sample_token]
        ego_pose = ego_poses_dict.get(sample_token) if ego_poses_dict else None
        
        # Update tracker and get tracked objects
        tracked_objects = tracker.update(detections, timestamp, ego_pose)
        
        # Convert to natural language
        nl_desc = format_tracked_objects_to_nl(tracked_objects)
        nl_descriptions[sample_token] = nl_desc
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            for sample_token, timestamp in sorted_samples:
                if sample_token in nl_descriptions:
                    f.write(f"Sample Token: {sample_token} (Time: {timestamp:.3f}s)\n")
                    f.write(nl_descriptions[sample_token])
                    f.write("\n\n" + "="*80 + "\n\n")
    
    return nl_descriptions


# Example usage
if __name__ == "__main__":
    # Load your results
    with open("results_nusc.json", 'r') as f:
        results_dict = json.load(f)
    
    # You need to provide timestamps for each sample
    # Example: extract from nuscenes dataset
    from nuscenes.nuscenes import NuScenes
    
    # Initialize nuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/yl817/BEVFormer/data/nuscenes', verbose=False)
    
    # Build timestamp and ego pose dictionaries
    timestamps_dict = {}
    ego_poses_dict = {}
    
    for sample_token in results_dict['results'].keys():
        sample = nusc.get('sample', sample_token)
        timestamps_dict[sample_token] = sample['timestamp'] / 1e6  # Convert to seconds
        
        # Get ego pose
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_poses_dict[sample_token] = {
            'translation': ego_pose_record['translation'],
            'rotation': ego_pose_record['rotation']
        }
    
    # Process all samples with tracking
    nl_descriptions = process_sequential_results(
        results_dict['results'],
        timestamps_dict,
        ego_poses_dict,
        output_file="detection_descriptions_with_tracking.txt"
    )
    
    # Print example
    first_token = sorted(timestamps_dict.items(), key=lambda x: x[1])[0][0]
    print(f"Example output for sample {first_token}:")
    print(nl_descriptions[first_token])