"""
Generate synthetic physics videos for training the world model.

Creates simple physics simulations:
- Bouncing ball
- Pendulum
- Falling object
- Projectile motion
"""
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


class BouncingBall:
    """Simulate a bouncing ball with gravity."""
    
    def __init__(self, width=64, height=64, ball_radius=16):
        self.width = width
        self.height = height
        self.ball_radius = ball_radius
        
        # Physics parameters
        self.gravity = 0.5
        self.damping = 0.85  # Energy loss on bounce
        
        # Random initial position and velocity
        self.x = np.random.uniform(ball_radius + 5, width - ball_radius - 5)
        self.y = np.random.uniform(ball_radius + 5, height // 2)
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-1, 1)
        
        # Random color
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
    
    def step(self):
        """Update physics for one timestep."""
        # Apply gravity
        self.vy += self.gravity
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off walls
        if self.x - self.ball_radius < 0 or self.x + self.ball_radius > self.width:
            self.vx = -self.vx * self.damping
            self.x = np.clip(self.x, self.ball_radius, self.width - self.ball_radius)
        
        if self.y + self.ball_radius > self.height:
            self.vy = -self.vy * self.damping
            self.y = self.height - self.ball_radius
            # Add small random bounce
            self.vx += np.random.uniform(-0.2, 0.2)
        
        if self.y - self.ball_radius < 0:
            self.vy = abs(self.vy) * self.damping
            self.y = self.ball_radius
    
    def render(self, frame):
        """Render the ball on the frame."""
        cv2.circle(frame, (int(self.x), int(self.y)), 
                   self.ball_radius, self.color, -1)
        return frame


class Pendulum:
    """Simulate a simple pendulum."""
    
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        
        # Pivot point
        self.pivot_x = width // 2
        self.pivot_y = 5
        
        # Pendulum parameters
        self.length = np.random.uniform(20, 35)
        self.angle = np.random.uniform(-np.pi/3, np.pi/3)
        self.angular_velocity = 0
        self.gravity = 0.5
        self.damping = 0.995
        
        # Random color
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
        self.bob_radius = 4
    
    def step(self):
        """Update physics for one timestep."""
        # Simple pendulum equation: α = -(g/L) * sin(θ)
        angular_acceleration = -(self.gravity / self.length) * np.sin(self.angle)
        
        self.angular_velocity += angular_acceleration
        self.angular_velocity *= self.damping
        self.angle += self.angular_velocity
    
    def render(self, frame):
        """Render the pendulum on the frame."""
        # Calculate bob position
        bob_x = int(self.pivot_x + self.length * np.sin(self.angle))
        bob_y = int(self.pivot_y + self.length * np.cos(self.angle))
        
        # Draw string
        cv2.line(frame, (self.pivot_x, self.pivot_y), (bob_x, bob_y),
                (200, 200, 200), 1)
        
        # Draw pivot
        cv2.circle(frame, (self.pivot_x, self.pivot_y), 2, (100, 100, 100), -1)
        
        # Draw bob
        cv2.circle(frame, (bob_x, bob_y), self.bob_radius, self.color, -1)
        
        return frame


class FallingObject:
    """Simulate a falling object with air resistance."""
    
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        
        # Random initial position (top half)
        self.x = np.random.uniform(10, width - 10)
        self.y = np.random.uniform(5, height // 3)
        
        # Physics
        self.vx = np.random.uniform(-1, 1)
        self.vy = 0
        self.gravity = 0.4
        self.air_resistance = 0.98
        self.damping = 0.7
        
        # Object properties
        self.size = np.random.randint(3, 6)
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
        self.shape = np.random.choice(['circle', 'square'])
    
    def step(self):
        """Update physics for one timestep."""
        # Apply gravity and air resistance
        self.vy += self.gravity
        self.vx *= self.air_resistance
        self.vy *= self.air_resistance
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off ground
        if self.y + self.size > self.height:
            self.vy = -self.vy * self.damping
            self.y = self.height - self.size
            # Reduce horizontal velocity on bounce
            self.vx *= 0.9
        
        # Bounce off walls
        if self.x - self.size < 0 or self.x + self.size > self.width:
            self.vx = -self.vx * self.damping
            self.x = np.clip(self.x, self.size, self.width - self.size)
    
    def render(self, frame):
        """Render the object on the frame."""
        if self.shape == 'circle':
            cv2.circle(frame, (int(self.x), int(self.y)), 
                      self.size, self.color, -1)
        else:  # square
            x1 = int(self.x - self.size)
            y1 = int(self.y - self.size)
            x2 = int(self.x + self.size)
            y2 = int(self.y + self.size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        
        return frame


class ProjectileMotion:
    """Simulate projectile motion."""
    
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        
        # Launch from bottom
        self.x = np.random.uniform(5, width // 3)
        self.y = height - 5
        
        # Initial velocity
        launch_angle = np.random.uniform(np.pi/6, np.pi/3)
        launch_speed = np.random.uniform(3, 5)
        self.vx = launch_speed * np.cos(launch_angle)
        self.vy = -launch_speed * np.sin(launch_angle)
        
        # Physics
        self.gravity = 0.3
        
        # Object properties
        self.radius = 3
        self.color = tuple(np.random.randint(100, 255, 3).tolist())
        self.trail = []  # Store recent positions for trail effect
        self.max_trail = 5
    
    def step(self):
        """Update physics for one timestep."""
        # Store position for trail
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        
        # Apply gravity
        self.vy += self.gravity
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off ground
        if self.y + self.radius > self.height:
            self.vy = -self.vy * 0.7
            self.y = self.height - self.radius
        
        # Bounce off walls
        if self.x - self.radius < 0 or self.x + self.radius > self.width:
            self.vx = -self.vx * 0.8
            self.x = np.clip(self.x, self.radius, self.width - self.radius)
    
    def render(self, frame):
        """Render the projectile with trail."""
        # Draw trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail)
            radius = max(1, int(self.radius * alpha * 0.5))
            color = tuple(int(c * alpha * 0.5) for c in self.color)
            cv2.circle(frame, (tx, ty), radius, color, -1)
        
        # Draw projectile
        cv2.circle(frame, (int(self.x), int(self.y)), 
                  self.radius, self.color, -1)
        
        return frame


def check_ball_collision(ball1, ball2):
    """Check and resolve collision between two balls."""
    dx = ball2.x - ball1.x
    dy = ball2.y - ball1.y
    distance = np.sqrt(dx**2 + dy**2)
    min_distance = ball1.ball_radius + ball2.ball_radius
    
    if distance < min_distance and distance > 0:
        # Collision detected - exchange velocities (elastic collision)
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        dvx = ball1.vx - ball2.vx
        dvy = ball1.vy - ball2.vy
        
        # Relative velocity in collision normal direction
        dvn = dvx * nx + dvy * ny
        
        # Only resolve if objects are moving toward each other
        if dvn > 0:
            # Update velocities (assuming equal mass)
            ball1.vx -= dvn * nx
            ball1.vy -= dvn * ny
            ball2.vx += dvn * nx
            ball2.vy += dvn * ny
            
            # Separate balls to avoid overlap
            overlap = min_distance - distance
            ball1.x -= overlap * 0.5 * nx
            ball1.y -= overlap * 0.5 * ny
            ball2.x += overlap * 0.5 * nx
            ball2.y += overlap * 0.5 * ny


def generate_video(simulation_type, num_frames=300, width=64, height=64, 
                   num_objects=1, fps=30, output_path=None):
    """
    Generate a physics simulation video.
    
    Args:
        simulation_type: Type of simulation ('bouncing', 'pendulum', 'falling', 'projectile')
        num_frames: Number of frames to generate
        width: Frame width
        height: Frame height
        num_objects: Number of objects to simulate
        fps: Frames per second (for saving)
        output_path: Path to save video (if None, returns frames)
    
    Returns:
        frames: numpy array of shape (num_frames, height, width, 3)
    """
    # Create simulation objects
    objects = []
    for _ in range(num_objects):
        if simulation_type == 'bouncing':
            objects.append(BouncingBall(width, height))
        elif simulation_type == 'pendulum':
            objects.append(Pendulum(width, height))
        elif simulation_type == 'falling':
            objects.append(FallingObject(width, height))
        elif simulation_type == 'projectile':
            objects.append(ProjectileMotion(width, height))
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
    
    # Generate frames
    frames = []
    for _ in range(num_frames):
        # Create blank frame (white background)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Update each object
        for obj in objects:
            obj.step()
        
        # Check collisions between bouncing balls
        if simulation_type == 'bouncing' and num_objects > 1:
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    check_ball_collision(objects[i], objects[j])
        
        # Render each object
        for obj in objects:
            frame = obj.render(frame)
        
        frames.append(frame)
    
    frames = np.array(frames)
    
    # Save video if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    return frames


def generate_dataset(output_dir, num_videos_per_type=10, num_frames=100,
                    width=64, height=64, fps=30):
    """
    Generate a complete dataset of physics videos.
    
    Args:
        output_dir: Directory to save videos
        num_videos_per_type: Number of videos per simulation type
        num_frames: Frames per video
        width: Frame width
        height: Frame height
        fps: Frames per second
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    #simulation_types = ['bouncing', 'pendulum', 'falling', 'projectile']
    simulation_types = ['bouncing']
    total_videos = num_videos_per_type * len(simulation_types)
    
    print(f"Generating {total_videos} physics videos...")
    print(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {fps}")
    
    with tqdm(total=total_videos) as pbar:
        for sim_type in simulation_types:
            # Create subdirectory for each type
            type_dir = output_path / sim_type
            type_dir.mkdir(exist_ok=True)
            
            for i in range(num_videos_per_type):
                # Random number of objects (1-5 for bouncing, 1-3 for falling, 1 for others)
                if sim_type == 'bouncing':
                    num_objects = np.random.randint(1, 9)  # 1-5 balls
                    video_width = width
                    video_height = height
                elif sim_type == 'falling':
                    num_objects = np.random.randint(1, 9)  # 1-3 objects
                    video_width = width
                    video_height = height
                else:
                    num_objects = 1
                    video_width = width
                    video_height = height
                
                video_path = type_dir / f"{sim_type}_{i:03d}.mp4"
                
                generate_video(
                    sim_type,
                    num_frames=num_frames,
                    width=video_width,
                    height=video_height,
                    num_objects=num_objects,
                    fps=fps,
                    output_path=str(video_path)
                )
                
                pbar.update(1)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Saved to: {output_dir}")
    print(f"📊 Total videos: {total_videos}")
    print(f"   - Bouncing balls: {num_videos_per_type}")
    print(f"   - Pendulums: {num_videos_per_type}")
    print(f"   - Falling objects: {num_videos_per_type}")
    print(f"   - Projectiles: {num_videos_per_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic physics videos for world model training"
    )
    parser.add_argument("--output", type=str, default="data/raw",
                       help="Output directory for videos")
    parser.add_argument("--num-videos", type=int, default=512,
                       help="Number of videos per simulation type")
    parser.add_argument("--num-frames", type=int, default=300,
                       help="Number of frames per video")
    parser.add_argument("--width", type=int, default=256,
                       help="Frame width")
    parser.add_argument("--height", type=int, default=256,
                       help="Frame height")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    
    args = parser.parse_args()
    
    generate_dataset(
        args.output,
        num_videos_per_type=args.num_videos,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
