import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time 

class HeatConduction2D:
    """
    A class to simulate steady-state heat conduction in a 2-dimensional rectangular domain.

    Attributes:
        M (int): Number of grid points in the x-direction.
        N (int): Number of grid points in the y-direction.
        width (float): Width of the domain.
        height (float): Height of the domain.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        T_top (float): Temperature at the top boundary.
        T_bottom (float): Temperature at the bottom boundary.
        T_left (float): Temperature at the left boundary.
        T_right (float): Temperature at the right boundary.
        A (lil_matrix): Sparse matrix representing the discretized domain.
        b (np.array): Right-hand side vector for the linear system.
        vectorized (bool): Flag for using vectorized operations.

    Methods:
        calculate_boundary_conditions(): Sets up the boundary conditions.
        calculate_interior_points_vectorized(): Fills in the matrix `A` for interior points using vectorized operations.
        calculate_interior_points(): Fills in the matrix `A` for interior points using loop-based operations.
        solve(): Solves the linear system to find the temperature distribution.
        plot(): Plots the temperature distribution over the domain.
    """
    def __init__(self, M, N, width, height, T_top, T_bottom, T_left, T_right, vectorized=True):
        """
        Initializes the HeatConduction2D class with domain parameters and boundary conditions.

        Parameters:
            M, N (int): Number of grid points in the x and y directions, respectively.
            width, height (float): Physical dimensions of the rectangular domain.
            T_top, T_bottom, T_left, T_right (float): Temperatures at the domain boundaries.
            vectorized (bool, optional): If True, use vectorized operations for interior points calculation. Defaults to True.
        """
        self.M, self.N = M, N
        self.width, self.height = width, height
        self.dx = self.width / (self.M - 1)
        self.dy = self.height / (self.N - 1)
        self.T_top, self.T_bottom, self.T_left, self.T_right = T_top, T_bottom, T_left, T_right
        self.A = lil_matrix((self.M*self.N, self.M*self.N))
        self.b = np.zeros(self.M*self.N)
        self.calculate_boundary_conditions()
        self.calculate_interior_points_vectorized() if vectorized else self.calculate_interior_points()

    def calculate_boundary_conditions(self):
        """
        Sets up boundary conditions by modifying the `A` matrix and `b` vector 
        to impose fixed temperatures at the domain's edges.
        """
        start_time = time.time()   
        bottom_boundary = np.arange(0, self.N)
        top_boundary = np.arange((self.M-1)*self.N, self.M*self.N)
        left_boundary = np.arange(0, self.M*self.N, self.N)
        right_boundary = np.arange(self.N-1, self.M*self.N, self.N)

        self.A[bottom_boundary, bottom_boundary] = 1
        self.b[bottom_boundary] = self.T_bottom

        self.A[top_boundary, top_boundary] = 1
        self.b[top_boundary] = self.T_top

        self.A[left_boundary, left_boundary] = 1
        self.b[left_boundary] = self.T_left

        self.A[right_boundary, right_boundary] = 1
        self.b[right_boundary] = self.T_right

        self.boundary_points = np.concatenate([bottom_boundary, top_boundary, left_boundary, right_boundary])
        print(f'Boundary conditions calculated in {time.time() - start_time:.6f} seconds')
  
    def calculate_interior_points(self):
        """
        Computes coefficients for interior grid points using a non-vectorized approach.
        
        This method fills the sparse matrix A with coefficients corresponding to the discrete Laplacian
        operator applied to interior grid points, according to the finite difference method. It uses
        loop-based approach to iterate over grid points.
        """
        start_time = time.time()    
        interior_points = np.setdiff1d(np.arange(self.M*self.N), self.boundary_points)
        for point in interior_points:
            self.A[point, point] = -2 * (1/self.dx**2 + 1/self.dy**2)

            # Left neighbor, check if not first column unless it's a left boundary itself
            if (point % self.N) != 0:
                self.A[point, point-1] = 1/self.dx**2

            # Right neighbor, check if not last column unless it's a right boundary itself
            if (point % self.N) != (self.N - 1):
                self.A[point, point+1] = 1/self.dx**2

            # Bottom neighbor, simply check if not in the first row
            if point - self.N >= 0:
                self.A[point, point-self.N] = 1/self.dy**2

            # Top neighbor, simply check if not in the last row
            if point + self.N < self.M * self.N:
                self.A[point, point+self.N] = 1/self.dy**2
         
        print(f'Interior points calculated in {time.time() - start_time:.6f} seconds')

             
    def solve(self):
        """
        Solves the linear system to find the steady-state temperature distribution across the domain.

        This method converts the `A` matrix to CSR format for efficient solving, then uses `spsolve` from SciPy to solve the system. The resulting temperature vector is reshaped into a grid matching the domain layout.
        """
        start_time = time.time()
        self.A_csr = self.A.tocsr()
        self.T = spsolve(self.A_csr, self.b)
        print(f'Solution calculated in {time.time() - start_time:.6f} seconds')
        self.T_grid = self.T.reshape((self.M, self.N))
        return self.T_grid
    
    def plot(self):
        """
        Plots the temperature distribution over the domain.

        Generates a heatmap of the temperature distribution using matplotlib, with the temperature scale represented as a color gradient. Axes are labeled with physical dimensions.
        """
        fig = plt.figure(figsize=(8, 6), dpi=200)
        plt.imshow(self.T_grid, origin='lower', extent=[0, self.width, 0, self.height])
        plt.colorbar(label='Temperature')
        plt.xlabel('Width (m)')
        plt.ylabel('Height (m)')
        plt.title('2D Steady-State Heat Conduction')
        plt.show()

class HeatConduction2DTransient:
    """
    A class to simulate transient heat conduction in a 2-dimensional rectangular domain.

    Attributes:
        M, N (int): Number of grid points in the x and y directions, respectively. For a square domain, M = N.
        width, height (float): Dimensions of the domain. For a square domain, width = height.
        dx (float): Grid spacing, equal in both the x and y directions due to the square nature of the domain.
        T_init (float): Initial uniform temperature of the domain.
        T_top (float): Temperature at the top boundary.
        T_bottom (float): Temperature at the bottom boundary.
        T_left (float): Temperature at the left boundary.
        T_right (float): Temperature at the right boundary.
        alpha (float): Thermal diffusivity of the material.
        dt (float): Time step size.
        time_steps (int): Total number of time steps to simulate.
        A (lil_matrix): Sparse matrix used in the Crank-Nicolson scheme for time step n+1.
        B (lil_matrix): Sparse matrix used in the Crank-Nicolson scheme for time step n.
        b (np.array): Right-hand side vector for boundary conditions.
        T (np.array): Current temperature distribution in the domain.
        centerline_temps (list): Recorded temperatures along the centerline over time.
        evolution (list): Snapshots of the temperature distribution at various time steps.

    Methods:
        calculate_matrices(): Sets up the A and B matrices for the Crank-Nicolson scheme.
        calculate_boundary_conditions(): Defines the boundary conditions for the simulation.
        solve(): Solves for the temperature distribution over the specified time steps.
        track_temperature(): Records the temperature at the domain's centerline at each time step.
        plot_centerline(num_of_points=8): Plots the evolution of centerline temperature.
        plot_temperature_evolution(num_of_points=16): Visualizes the temperature distribution at selected time steps.
    """

    def __init__(self, M, N, width, height, T_init, T_top, T_bottom, T_left, T_right, alpha, dt, time_steps):
        self.M, self.N = M, N
        self.width, self.height = width, height
        self.dx = self.width / (self.M - 1)
        self.dy = self.height / (self.N - 1)
        self.T_init = T_init
        self.T_top, self.T_bottom, self.T_left, self.T_right = T_top, T_bottom, T_left, T_right
        self.alpha = alpha
        self.dt = dt
        self.time_steps = int(time_steps/self.dt)
        
        self.A = lil_matrix((self.M*self.N, self.M*self.N))
        self.B = lil_matrix((self.M*self.N, self.M*self.N))
        self.b = np.zeros(self.M*self.N)
        self.T = np.ones(self.M*self.N) * T_init
        self.centerline_temps = []
        self.evolution = []
        
        self.calculate_boundary_conditions()
        self.calculate_matrices()

    def calculate_matrices(self):
        D = self.alpha * self.dt / (2 * self.dx**2)  # Assuming dx = dy
        # Set up A matrix for time step n+1
        for i in range(1, self.M-1):
            for j in range(1, self.N-1):
                index = i*self.N + j
                self.A[index, index] = 1 + 4*D
                self.A[index, index-1] = -D
                self.A[index, index+1] = -D
                self.A[index, index-self.N] = -D
                self.A[index, index+self.N] = -D

        # Set up B matrix for time step n
        for i in range(1, self.M-1):
            for j in range(1, self.N-1):
                index = i*self.N + j
                self.B[index, index] = 1 - 4*D
                self.B[index, index-1] = D
                self.B[index, index+1] = D
                self.B[index, index-self.N] = D
                self.B[index, index+self.N] = D

    # Other methods (calculate_boundary_conditions, solve, etc.) remain the same
    def calculate_boundary_conditions(self):
        bottom_boundary = np.arange(0, self.N)
        top_boundary = np.arange((self.M-1)*self.N, self.M*self.N)
        left_boundary = np.arange(0, self.M*self.N, self.N)
        right_boundary = np.arange(self.N-1, self.M*self.N, self.N)

        self.A[bottom_boundary, bottom_boundary] = 1
        self.b[bottom_boundary] = self.T_bottom

        self.A[top_boundary, top_boundary] = 1
        self.b[top_boundary] = self.T_top

        self.A[left_boundary, left_boundary] = 1
        self.b[left_boundary] = self.T_left

        self.A[right_boundary, right_boundary] = 1
        self.b[right_boundary] = self.T_right

        self.boundary_points = np.concatenate([bottom_boundary, top_boundary, left_boundary, right_boundary])
    
    def solve(self):
        self.A_csr = self.A.tocsr()
        self.B_csr = self.B.tocsr()
        for step in range(self.time_steps):
            # Apply boundary conditions if they change over time
            self.T = spsolve(self.A_csr, self.B_csr.dot(self.T) + self.b)
            self.track_temperature()

        print(f'Solution calculated over {self.time_steps} time steps.')
    
    def solve_forward_euler(self):
        for step in range(time_steps):
            T_new = T.copy()
            for i in range(1, M-1):
                for j in range(1, N-1):
                    T_new[i, j] = T[i, j] + self.alpha * self.dt * (
                        (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / self.dx**2 +
                        (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / self.dy**2)
            T = T_new

            self.T = T.copy()
            self.track_temperature()

        print(f'Solution calculated over {self.time_steps} time steps using Forward Euler.')
        
    def track_temperature(self):
        temp_grid = self.T.reshape((self.M, self.N))
        self.evolution.append(temp_grid)
        center_index = self.M // 2
        self.centerline_temps.append(temp_grid[center_index, :])
        
    def plot_centerline(self, num_of_points=8):
        fig = plt.figure(figsize=(8, 6), dpi=200)
        num_of_points = num_of_points
        points = self.time_steps // num_of_points
        centerline_temp = np.array(self.centerline_temps)
        for i in range(0, self.time_steps, points):
            plt.plot(centerline_temp[i], label=f'Time step {i}')
        plt.xlabel('x')
        plt.ylabel('Temperature')
        plt.title('Centerline Temperature Evolution')
        plt.legend()
        plt.show()
        
    def plot_temperature_evolution(self, num_of_points=16):
        fig = plt.figure(figsize=(8, 6), dpi=200)
        num_of_points = num_of_points
        points = self.time_steps // num_of_points
        for i in range(0, self.time_steps, points):
            plt.subplot(4, 4, i//points + 1)
            plt.imshow(self.evolution[i], origin='lower', extent=[0, self.width, 0, self.height])
            plt.colorbar(label='Temperature')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Time step {i}')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()


        
