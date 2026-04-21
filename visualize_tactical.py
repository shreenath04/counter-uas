import time
import numpy as np
import pyvista as pv
from stable_baselines3 import PPO
from sb3_env import CounterUASGymEnv

# --- CONFIGURATION ---
MODEL_PATH = "ppo_counter_uas_best_v2_ft" 
GRID_SIZE = 50
RENDER_FPS = 20  
MAX_TRAIL_LEN = 5  

# --- TACTICAL COLORS (RGB 0.0 to 1.0) ---
RGB_GROUND = (0.05, 0.05, 0.05)
RGB_BASE = (0.17, 0.24, 0.31)
RGB_FRIENDLY = (0.0, 0.82, 1.0)
RGB_HOSTILE = (1.0, 0.2, 0.18)
RGB_INTERCEPT = (0.0, 1.0, 0.5)

class TacticalVisualizer:
    def __init__(self, env):
        self.env = env
        self.num_friendly = 5
        self.num_hostile = 5 
        
        self.plotter = pv.Plotter(title="Counter-UAS: Tactical Command")
        self.plotter.set_background(RGB_GROUND)
        
        # Ground
        ground = pv.Plane(center=(25, 25, 0), direction=(0, 0, 1), i_size=75, j_size=75)
        self.plotter.add_mesh(ground, color=(0.1, 0.1, 0.1), specular=0.1)
        
        # Base Cube aka Command Center
        base_cube = pv.Cube(center=(25, 25, 2), x_length=4, y_length=4, z_length=4)
        self.plotter.add_mesh(base_cube, color=RGB_BASE, opacity=0.8, label='Command Center')
        
        # Engagement Sphere
        base_sphere = pv.Sphere(radius=12, center=(25, 25, 0))
        self.plotter.add_mesh(base_sphere, color=RGB_BASE, opacity=0.15, style='wireframe')

        # Engagement Sphere 2
        base_sphere = pv.Sphere(radius=30, center=(25, 25, 0))
        self.plotter.add_mesh(base_sphere, color=RGB_BASE, opacity=0.10, style='wireframe')
        
        # Drones
        self.friendly_meshes = []
        for i in range(self.num_friendly):
            a = self.plotter.add_mesh(pv.Sphere(radius=1.0), color=RGB_FRIENDLY)
            self.friendly_meshes.append(a)
            
        self.hostile_meshes = []
        for i in range(self.num_hostile):
            a = self.plotter.add_mesh(pv.Sphere(radius=1.2), color=RGB_HOSTILE)
            self.hostile_meshes.append(a)

        self.friendly_trails = [[] for _ in range(self.num_friendly)]
        self.plotter.camera_position = [(120, 120, 80), (25, 25, 0), (0, 0, 1)]

    def update_trails(self, idx, points_list):
        name = f"trail_{idx}"
        self.plotter.remove_actor(name)
        if len(points_list) < 2: return
        pts = np.array(points_list)
        cells = np.hstack([np.full((len(pts)-1, 1), 2), np.arange(len(pts)-1)[:, None], np.arange(1, len(pts))[:, None]]).flatten()
        line = pv.PolyData(pts)
        line.lines = cells
        self.plotter.add_mesh(line, color=RGB_FRIENDLY, line_width=3, opacity=0.3, name=name)

    def render_step(self):
        # Reach deep into the nested env structure
        core = self.env.env 

        for i in range(self.num_friendly):
            if core.friendly_alive[i]:
                pos = core.friendly_drones[i]
                self.friendly_meshes[i].SetPosition(pos)
                self.friendly_meshes[i].GetProperty().SetColor(RGB_FRIENDLY)
                self.friendly_trails[i].append(pos)
                if len(self.friendly_trails[i]) > MAX_TRAIL_LEN: self.friendly_trails[i].pop(0)
                self.update_trails(i, self.friendly_trails[i])
            else:
                self.friendly_meshes[i].GetProperty().SetColor(RGB_INTERCEPT)

        for i in range(self.num_hostile):
            if core.hostile_alive[i]:
                self.hostile_meshes[i].SetPosition(core.hostile_drones[i])
            else:
                self.hostile_meshes[i].SetPosition((0, 0, -10)) 
        
        self.plotter.render()

def main():
    env = CounterUASGymEnv()
    model = PPO.load(MODEL_PATH)
    viz = TacticalVisualizer(env)
    
    obs, _ = env.reset()
    viz.plotter.show(interactive_update=True)
    
    print("Tactical Simulation Active. Close window or Ctrl+C to stop.")
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            viz.render_step()
            viz.plotter.update() 
            
            time.sleep(1.0 / RENDER_FPS)
            if done:
                print(f"Intercepted: {info.get('intercepted', 'N/A')} | Breaches: {info.get('breaches', 'N/A')}")
                time.sleep(1.2)
                obs, _ = env.reset()
                for t in viz.friendly_trails: t.clear()
    except Exception as e:
        print(f"Exiting: {e}")
    finally:
        viz.plotter.close()

if __name__ == "__main__":
    main()