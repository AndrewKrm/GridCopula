import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
import pickle
from scipy.stats import norm
from abc import ABC, abstractmethod

class GridCopula(ABC):
    def __init__(self, m, n, input):
        self.m = m
        self.n = n
        self.width = 1 / (self.m - 1)
        self.grid= self.optimize_grid(input)
        self.dim1 = 0
        self.dim2 = 1
        self.projected_grid = self.project_to_dims([self.dim1, self.dim2])
        self.projected_grid_volume = self.compute_hypercubes_volume(self.projected_grid)
        self.hypercubes_volume = self.compute_hypercubes_volume()
    
    @abstractmethod
    def optimize_grid(self, input):
        pass

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def set_projected_dimensions(self, dim1, dim2, cond=None):
        self.dim1 = dim1
        self.dim2 = dim2
        self.cond = cond
        if cond is not None:
            self.projected_grid = self.project_to_dims_cond([dim1, dim2])
            self.projected_grid_volume = self.compute_hypercubes_volume(self.projected_grid)
        else:
            self.projected_grid = self.project_to_dims([dim1, dim2])
            self.projected_grid_volume = self.compute_hypercubes_volume(self.projected_grid)

    def compute_hypercubes_volume(self, data=None):
        # Find the volume of each cells in the 
        if data is not None:
            def sum_corners(idx):
                return sum(data[tuple(np.array(idx) + corner)] for corner in np.ndindex((2,)*data.ndim)) * ((self.width/2) ** data.ndim)
            
            cell_volume = np.zeros((self.m-1,)*data.ndim)
            indices = np.indices((self.m-1,)*data.ndim)

            
            for idx in indices.reshape(data.ndim, -1).T:
                cell_volume[tuple(idx)] = sum_corners(tuple(idx))
            return cell_volume
        else:
            def sum_corners(idx):
                return sum(self.grid[tuple(np.array(idx) + corner)] for corner in np.ndindex((2,)*self.n)) * ((self.width/2) ** self.n)
            cell_volume = np.zeros((self.m-1,)*self.n)
            indices = np.indices((self.m-1,)*self.n)

            
            for idx in indices.reshape(self.n, -1).T:
                cell_volume[tuple(idx)] = sum_corners(tuple(idx))
            return cell_volume
        
    def normalize_volume(self, data):
        return data/np.sum(self.compute_hypercubes_volume(data))

    def create_grid_weights(self, dims):
        def calculate_weights(n, m):
            # Initialize the grid
            grid = np.zeros([m]*n, dtype=int)
            # Helper function to determine the number of contacts for a given point
            def num_contacts(point):
                contacts = 1
                for coord in point:
                    if coord == 0 or coord == m-1:
                        contacts *= 2
                    else:
                        contacts *= 4
                return contacts // (2 ** n)
            # Iterate through each point in the grid
            it = np.nditer(grid, flags=['multi_index'], op_flags=['writeonly'])
            while not it.finished:
                it[0] = num_contacts(it.multi_index)
                it.iternext()
            return grid
        def replicate_arr(n_arr, dims):
            # Initial replication across the specified dimension
            replicated_arr = np.expand_dims(n_arr, axis=(dims))
            
            # Calculate the replication factor for each dimension
            replication_factors = [1] * self.n

            for dim in dims:
                replication_factors[dim] = self.m

            # Replicate across the specified dimension
            replicated_arr = np.tile(replicated_arr, replication_factors)
            
            return replicated_arr
        
        return replicate_arr(calculate_weights(self.n-len(dims), self.m),dims)

    def project_to_dims(self, dims):
        integral=self.create_grid_weights(dims)
        projected_grid=np.multiply(self.grid, integral).sum(axis=tuple(i for i in range(self.n) if i not in dims))
        return projected_grid*((self.width/2)**(self.n-len(dims)))

    def project_to_dims_cond(self, dims):
        # find the index in self.cond for which the values are not -1 and count how many of the dims are smaller than this index
        dims = sorted(dims + [i for i, x in enumerate(self.cond) if x != -1])

        integral=self.create_grid_weights(dims)
        projected_grid=np.multiply(self.grid, integral).sum(axis=tuple(i for i in range(self.n) if i not in dims))
        projected_grid*=((self.width/2)**(self.n-len(dims)))
        for i, c in enumerate(self.cond):
            if c != -1:
                c_index = c * (self.m - 1)
                c0 = int(c_index)
                if c0 == self.m - 1:
                    c0 -= 1
                c1 = min(c0 + 1, self.m - 1)
                dc = c_index - c0
                #create a slice for the dimension of cond to interpolate 
                grid_c0 = np.take(projected_grid, c0, axis=2)
                grid_c1 = np.take(projected_grid, c1, axis=2)
                projected_grid = (1 - dc) * grid_c0 + dc * grid_c1
        return projected_grid

    def show(self, mode='pdf'):
        
        xpos, ypos = np.meshgrid(
            np.linspace(0, 1, self.m),
            np.linspace(0, 1, self.m)
        )
        zpos = np.zeros_like(xpos)

        # Apply the interpolator to each (x, y) pair
        for i in range(self.m):
            for j in range(self.m):
                if mode == 'pdf':
                    zpos[i, j] = self.projected_grid[i, j]
                elif mode == 'cdf':
                    zpos[i, j] = self.cdf(xpos[i, j], ypos[i, j])
                else:
                    raise ValueError("Invalid mode. Expected 'pdf' or 'cdf'.")

        # Use Plotly for 3D plotting
        fig = go.Figure(data=[go.Surface(z=zpos, x=xpos, y=ypos)])

        fig.update_layout(title=f'3D Plot of bivariate Copula {mode} with {self.dim1} and {self.dim2} as dimensions', autosize=True,
                        scene=dict(
                            xaxis_title='X coordinate',
                            yaxis_title='Y coordinate',
                            zaxis_title='Probability',
                            aspectmode='manual',
                            aspectratio=dict(x=1, y=1, z=1)
                        ),
                        margin=dict(l=65, r=50, b=65, t=90))

        # Show the plot
        fig.show()
    
    def cdf(self, x, y):
        x_index = x * (self.m - 1)
        y_index = y * (self.m - 1)
        x0, y0 = int(x_index), int(y_index)
        if x0 == self.m - 1:
            x0 -= 1
        if y0 == self.m - 1:
            y0 -= 1
        x1, y1 = min(x0 + 1, self.m - 1), min(y0 + 1, self.m - 1)
        dx, dy = x_index - x0, y_index - y0
        cell_integrals=np.zeros((x1,y1))
        cell_integrals[:x0,:y0]=self.projected_grid_volume[:x0,:y0]

        for i in range(x0):
            cell_integrals[i,y0]=((self.projected_grid[i,y0]*(2-dy)+self.projected_grid[i,y1]*dy)+(self.projected_grid[i+1,y0]*(2-dy)+self.projected_grid[i+1,y1]*dy))*dy*self.width**2/4

        for j in range(y0):
            cell_integrals[x0,j]=((self.projected_grid[x0,j]*(2-dx)+self.projected_grid[x1,j]*dx)+(self.projected_grid[x0,j+1]*(2-dx)+self.projected_grid[x1,j+1]*dx))*dx*self.width**2/4

        cell_integrals[x0,y0]=((self.projected_grid[y0, x0] * (1 - dx) * (1 - dy) +
                 self.projected_grid[y1, x0] * (1 - dx) * dy +
                 self.projected_grid[y0, x1] * dx * (1 - dy) +
                 self.projected_grid[y1, x1] * dx * dy)+self.projected_grid[y0, x0]+(self.projected_grid[x0,y0]*(1-dx)+self.projected_grid[x1,y0]*dx)+(self.projected_grid[x0,y0]*(1-dy)+self.projected_grid[x0,y1]*dy))*dx*dy*self.width**2/4
        
        return np.sum(cell_integrals)

class GridCopulaData(GridCopula):

    def __init__(self, data_points, m):
        super().__init__(m)
        self.std_dev = self.width
        self.grid = self.optimize_grid(self.normalize_volume(self.count_points_with_gaussian(data_points)))

    def count_points_with_gaussian(self, data_points):
        def gaussian_nd(x, mu, sigma):
            return np.exp(-np.sum((x - mu)**2) / (2 * sigma**2 * self.n))
        grid = np.zeros((self.m,) * self.n)  # n-dimensional grid
        cell_size = 1.0 / (self.m - 1)
        for data_point in data_points:
            # Calculate the range of grid points to iterate over for this data point
            min_indices = np.maximum((data_point - 4*self.std_dev) // cell_size, 0).astype(int)
            max_indices = np.minimum((data_point + 4*self.std_dev) // cell_size, self.m - 1).astype(int) + 1
            # Iterate over a hypercube around the data point
            for indices in np.ndindex(tuple(max_indices - min_indices)):
                indices = indices + min_indices  # Convert from relative to absolute indices
                grid_point = indices * cell_size
                grid[tuple(indices)] += gaussian_nd(grid_point, data_point, self.std_dev)
        return grid

    def optimize_grid(self, input_grid):

        half_width = self.width / 2
        # Create a variable grid of the same size as the input grid
        x = cp.Variable((self.m,)*self.n, nonneg=True)

        # Objective: Minimize the sum of squared differences from the input grid
        objective = cp.Minimize(cp.sum_squares(x - input_grid))

        # Constraints
        constraints = []

        for i in range(self.n):
            integral=self.create_grid_weights([i]).reshape(-1)
            for j in range(self.m):
                curr_coef=np.zeros((self.m,)*self.n)
                # Create a tuple of slices
                index = [slice(None)] * self.n
                # Replace the i-th slice with the index j
                index[i] = j
                curr_coef[tuple(index)] += 1
                const=cp.multiply(integral,curr_coef.reshape(-1))
                constraints.append(cp.sum(cp.multiply(x, const)) == (1/half_width)**(self.n-1))


        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return x.value
    
class GridCopulaEntropy(GridCopula):
    def __init__(self, spearman_rho, m):
        super().__init__(m, spearman_rho.shape[0], spearman_rho)
        self.spearman_rho = spearman_rho

    def optimize_grid(self, spearman_rho):

        # Define the objective
        x = cp.Variable((self.m**self.n), nonneg=True)
        
        def sum_over_dimensions(x, dim1, dim2):
            # Initialize an empty list to collect expressions
            expressions = []

            # Generate indices for the dimensions to be kept
            for i in range(self.m):
                for j in range(self.m):
                    # This list will hold expressions for the current element
                    element_expressions = []

                    # Iterate over all possible combinations for the sum dimensions
                    for sum_indices in np.ndindex(*(self.m for _ in range(self.n-2))):
                        # Construct the full N-dimensional index
                        idx = [0]*self.n
                        idx[dim1] = i
                        idx[dim2] = j
                        for k, sum_idx in enumerate(sum_indices, start=0):
                            if k >= dim1: k += 1
                            if k >= dim2: k += 1
                            idx[k] = sum_idx

                        # Convert the N-dimensional index to a flat index and add the corresponding x element
                        flat_index = np.ravel_multi_index(idx, [self.m]*self.n)
                        element_expressions.append(x[flat_index])

                    # Sum the expressions for the current element and add to the list
                    expressions.append(cp.sum(element_expressions))

            # Reshape the list of expressions into a 2D CVXPY expression
            result_expression = cp.reshape(cp.hstack(expressions), (self.m, self.m))

            return result_expression

        half_width = self.width / 2
        constraints = []

        entropy_weights = self.create_grid_weights([])

        for i in range(self.n):
            integral=self.create_grid_weights([i]).reshape(-1)
            for j in range(self.m):
                curr_coef=np.zeros((self.m,)*self.n)
                # Create a tuple of slices
                index = [slice(None)] * self.n
                # Replace the i-th slice with the index j
                index[i] = j
                curr_coef[tuple(index)] += 1
                const=cp.multiply(integral,curr_coef.reshape(-1))
                constraints.append(cp.sum(cp.multiply(x, const)) == (1/half_width)**(self.n-1))

        # Define constraints
        width_square = self.width ** 2
        c = np.zeros((self.m-1,self.m-1,4))
        for i in range(self.m-1):
            for j in range(self.m-1):
                a = i * self.width
                b = j * self.width
                c[i,j,0] = 1/3 * ((width_square)) * (3*a + self.width)   * (3*b + self.width)
                c[i,j,1] = 1/3 * ((width_square)) * (3*a + 2*self.width) * (3*b + self.width)
                c[i,j,2] = 1/3 * ((width_square)) * (3*a + self.width)   * (3*b + 2*self.width)
                c[i,j,3] = 1/3 * ((width_square)) * (3*a + 2*self.width) * (3*b + 2*self.width)

        for i in range(self.n-1):
            for j in range(i+1,self.n):
                integral=self.create_grid_weights([i,j]).reshape(-1)
                x_2=sum_over_dimensions(cp.multiply(x, integral),i,j)*((half_width)**(self.n-2))
                constraints.append(cp.sum(cp.multiply(x_2[:-1,:-1],c[:,:,0]) + cp.multiply(x_2[1:,:-1],c[:,:,1]) + cp.multiply(x_2[:-1,1:],c[:,:,2]) + cp.multiply(x_2[1:,1:],c[:,:,3])) - 3 == spearman_rho[i,j])

        objective = cp.Maximize(cp.sum(cp.multiply(cp.entr(x),entropy_weights.reshape(-1))))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=True, max_iter=1000)

        return (x.value).reshape((self.m,)*self.n)
