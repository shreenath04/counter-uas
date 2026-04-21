import numpy as np

class CounterUASEnv:
    def __init__(self, grid_size=50, num_friendly=5, num_hostile=5, max_steps=500):
        self.grid_size = grid_size
        self.num_friendly = num_friendly
        self.num_hostile = num_hostile
        self.max_steps = max_steps
        self.step_count = 0
        self.breaches = 0
        
        # Base position
        self.base = np.array([self.grid_size//2, self.grid_size//2, 0])
        
        # 27 possible actions: 3 choices per axis (-1, 0, +1) = 3^3
        self.actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    self.actions.append(np.array([dx, dy, dz]))
        
        # Zone boundaries (distance from base)
        self.zone_penalties = {1: -0.2, 2: -0.05, 3: -0.01}
        self.hostile_speed = 1.0
        
    def get_zone(self, position):
        dist = np.linalg.norm(position - self.base)
        if dist <= 12:
            return 1
        elif dist <= 30:
            return 2
        else:
            return 3
    
    def reset(self):
        self.step_count = 0
        self.breaches = 0
        
        # Friendly drones start near base
        self.friendly_drones = []
        self.friendly_alive = [True] * self.num_friendly
        for _ in range(self.num_friendly):
            pos = np.array([
                self.base[0] + np.random.randint(-3,4),
                self.base[1] + np.random.randint(-3,4),
                np.random.randint(0,3)
            ], dtype=float)
            self.friendly_drones.append(pos)
        
        # Hostile drones spawn at edges with random velocity toward base
        self.hostile_drones = []
        self.hostile_velocities = []
        self.hostile_alive = []
        
        for _ in range(self.num_hostile):
            pos = self._spawn_hostile()
            vel = self._compute_velocity_toward_base(pos)
            self.hostile_drones.append(pos)
            self.hostile_velocities.append(vel)
            self.hostile_alive.append(True)
        
        return self._get_state()
    
    def _spawn_hostile(self):
        # Spawn on a random face of the grid boundary
        while True:
            face = np.random.randint(5)
            pos = np.random.randint(0, self.grid_size, size=3).astype(float)
            
            if face == 0: pos[0] = self.grid_size - 1  
            elif face == 1: pos[0] = 0                  
            elif face == 2: pos[1] = self.grid_size - 1  
            elif face == 3: pos[1] = 0                  
            else: pos[2] = self.grid_size - 1
            
            # z >= 0 (no underground)
            pos[2] = max(pos[2], 1)
        
            if np.linalg.norm(pos - self.base) >= 30:
                return pos
    
    def _compute_velocity_toward_base(self, pos):
        direction = self.base - pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros(3)
        return (direction / norm)*self.hostile_speed  # unit vector toward base
    
    def _get_state(self):
        # Build state vector for each agent
        state = []
        for f in self.friendly_drones:
            state.extend(f)
        for i, h in enumerate(self.hostile_drones):
            state.extend(h)
            state.extend(self.hostile_velocities[i])
            state.append(float(self.hostile_alive[i]))
        return np.array(state, dtype=np.float32)
    
    def step(self, actions):
        # actions: list of action indices, one per friendly drone
        self.step_count += 1
        rewards = np.zeros(self.num_friendly)
        done = False
        
        # 1. Move friendly drones
        for i, action_idx in enumerate(actions):
            if not self.friendly_alive[i]:
                continue
            move = self.actions[action_idx]*1.5
            new_pos = self.friendly_drones[i] + move
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_pos[2] = max(new_pos[2], 0)
            self.friendly_drones[i] = new_pos
        
        # 2. Move hostile drones
        for i in range(self.num_hostile):
            if not self.hostile_alive[i]:
                continue
            self.hostile_drones[i] += self.hostile_velocities[i]
            self.hostile_drones[i] = np.clip(self.hostile_drones[i], 0, self.grid_size - 1)
        
        # 3. Check interceptions
        for i in range(self.num_friendly):
            if not self.friendly_alive[i]:
                continue
            for j in range(self.num_hostile):
                if not self.hostile_alive[j]:
                    continue
                dist = np.linalg.norm(self.friendly_drones[i] - self.hostile_drones[j])
                if dist <= 2.0:
                    self.hostile_alive[j] = False
                    self.friendly_alive[i] = False  # kamikaze

                    dist_from_base = np.linalg.norm(self.hostile_drones[j] - self.base)
                    max_dist = self.grid_size * 0.7
                    distance_bonus = 1.0 + (dist_from_base/max_dist)

                    rewards[i] += 3.0 * (distance_bonus*4.0)
                    for k in range(self.num_friendly):
                        if k != i:
                            rewards[k] += 1.5 * (distance_bonus*2.5)
                    break  # this drone is dead, stop checking other hostiles
        
        # 4. Check base breach
        for i in range(self.num_hostile):
            if not self.hostile_alive[i]:
                continue
            if np.linalg.norm(self.hostile_drones[i] - self.base) <= 6.0:
                rewards -= 8.0  # everyone penalized
                self.hostile_alive[i] = False
                self.breaches += 1
        
        # 5. Zone penalties for alive hostiles
        for i in range(self.num_hostile):
            if not self.hostile_alive[i]:
                continue
            zone = self.get_zone(self.hostile_drones[i])
            rewards += self.zone_penalties[zone]
        
        # 6. Friendly proximity penalty (repulsion)
        for i in range(self.num_friendly):
            for j in range(i + 1, self.num_friendly):
                dist = np.linalg.norm(self.friendly_drones[i] - self.friendly_drones[j])
                if dist <= 5.0:
                    penalty = -0.3 * (1 - dist / 5.0)  # stronger when closer
                    rewards[i] += penalty
                    rewards[j] += penalty
        
        # 7. Time penalty
        rewards -= 0.05

        # 8. Proximity reward + penalties
        for i in range(self.num_friendly):
            if not self.friendly_alive[i]:
                continue
            for j in range(self.num_hostile):
                if not self.hostile_alive[j]:
                    continue
                dist = np.linalg.norm(self.friendly_drones[i] - self.hostile_drones[j])
                dist_from_base = np.linalg.norm(self.hostile_drones[j]-self.base)


                if dist<=8.0 and dist_from_base<=12.0:
                    proximity_reward = -1.0*(1-dist/8.0)
                    rewards[i]+=proximity_reward

                if dist<=8.0 and dist_from_base<=30.0:
                    proximity_reward = -0.8*(1-dist/8.0)
                    rewards[i]+=proximity_reward

                if dist<=8.0:
                    proximity_reward = 0.03 * (1-dist/8.0)
                    rewards[i] += proximity_reward
                
                elif dist <= 30.0:
                    to_friendly = self.friendly_drones[i] - self.hostile_drones[j]
                    velocity_dir = self.hostile_velocities[j] / (np.linalg.norm(self.hostile_velocities[j]) + 1e-8)
                    alignment = np.dot(to_friendly, velocity_dir) / (dist + 1e-8)
                    
                    if alignment > 0:
                        proximity_reward = 0.01 * (1 - dist / 30.0) * alignment
                        rewards[i] += proximity_reward
        
        # 9. Check if done
        if all(not alive for alive in self.hostile_alive):
            done = True
        if all(not alive for alive in self.friendly_alive):
            done = True
        if self.step_count >= self.max_steps:
            done = True
        return self._get_state(), rewards, done


# Quick test
if __name__ == "__main__":
    env = CounterUASEnv()
    state = env.reset()
    print(f"State size: {len(state)}")
    print(f"Initial state: {state}")
    
    # Random actions for 2 friendly drones
    actions = [np.random.randint(27), np.random.randint(27)]
    next_state, rewards, done = env.step(actions)
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")