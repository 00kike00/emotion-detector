import numpy as np
import random

class APSO:
    def __init__(
        self,
        fitness_function,
        num_particles,
        num_dimensions,
        bounds,
        max_iterations=50,
        w=0.7,
        c1=2.0,
        c2=2.0,
        alpha=0.3
    ):
        """
        APSO optimizer

        Parameters:
        - fitness_function: function to minimize/maximize
        - num_particles: number of particles
        - num_dimensions: dimensionality of search space
        - bounds: (lower_bounds, upper_bounds)
        - max_iterations: iterations
        - w: inertia weight
        - c1: cognitive coefficient
        - c2: social coefficient
        - alpha: acceleration factor (APSO-specific)
        """

        self.fitness_function = fitness_function
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha

        self.lower_bounds = np.array(bounds[0])
        self.upper_bounds = np.array(bounds[1])

        self.max_velocity = (self.upper_bounds - self.lower_bounds) * 0.2

        # Initialize particles
        self.positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds,
            (num_particles, num_dimensions)
        )

        self.velocities = np.random.uniform(
            -1, 1, (num_particles, num_dimensions)
        )

        # Personal bests
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.array(
            [float('-inf')] * num_particles
        )

        # Global best
        self.gbest_position = None
        self.gbest_score = float('-inf')

    def optimize(self):
        for iteration in range(self.max_iterations):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration+1}/{self.max_iterations} | Current Best: {self.gbest_score:.2f}%")
            print(f"{'='*50}")
            for i in range(self.num_particles):

                # Evaluate fitness
                fitness = self.fitness_function(self.positions[i])

                # Update personal best
                if fitness > self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                # Update global best
                if fitness > self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i].copy()

            # Update particles
            for i in range(self.num_particles):

                r1 = random.random()
                r2 = random.random()

                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                acceleration = self.alpha * (self.gbest_position - self.pbest_positions[i])

                # Velocity update (APSO equation)
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + cognitive
                    + social
                    + acceleration
                )
                # Clamp velocity
                self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)

                # Position update
                self.positions[i] += self.velocities[i]

                # Apply bounds
                self.positions[i] = np.clip(
                    self.positions[i],
                    self.lower_bounds,
                    self.upper_bounds
                )

            # Optional: decrease inertia weight over time
            self.w *= 0.99

            print(f"Iteration {iteration+1}/{self.max_iterations} | Best Fitness: {self.gbest_score}")

        return self.gbest_position, self.gbest_score