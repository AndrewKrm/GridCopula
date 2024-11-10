import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
from scipy.stats import norm, spearmanr
from linear_entropy import linear_entr

class GridCopula:
    def __init__(self, m):
        self.m = m
        self.width = 1 / (m - 1)
        self.grid = np.ones((m, m))
        self.cell_volume = self.cell_volume_compute()
    
    def total_volume(self):
        self.cell_volume_compute()
        return np.sum(self.cell_volume)

    def cell_volume_compute(self):
        cell_volume = np.zeros((self.m-1,self.m-1))
        for i in range(self.m-1):
            for j in range(self.m-1):
                cell_volume[i,j] = (self.grid[i, j] + self.grid[i + 1, j] + self.grid[i, j + 1] + self.grid[i + 1, j + 1]) * (self.width ** 2) / 4
        return cell_volume
    
    def normalize_volume(self, input_grid):
        input_grid=input_grid/self.volume_grid(input_grid)
        return input_grid
    
    def volume_grid(self, input_grid):
        m = input_grid.shape[0]  # Assuming grid is m x m
        # Ensure the input is a NumPy array
        input_grid = np.array(input_grid)
        # The grid spacing h
        h = 1 / (m - 1)
        # Initialize the total volume
        total_volume = 0
        
        # Sum over each cell in the grid
        for i in range(m - 1):
            for j in range(m - 1):
                # Calculate the area for each cell
                cell_volume = (input_grid[i, j] + input_grid[i + 1, j] + input_grid[i, j + 1] + input_grid[i + 1, j + 1]) * (h ** 2) / 4
                # Add to the total area
                total_volume += cell_volume
        
        return total_volume
    
    def mirror_data(self, data_points, std_dev):
        copy_limit_corner=norm.ppf(0.99,scale=std_dev)
        copy_limit_border=copy_limit_corner*2/3
        mirrored_points = []
        for x, y in data_points:
            # Original point
            mirrored_points.append((x, y))

            # Mirror to the right
            if x > 1-copy_limit_border:
                mirrored_points.append((2 - x, y))

            # Mirror to the left
            if x < copy_limit_border:
                mirrored_points.append((-x, y))
            # Mirror to the top
            if y > 1-copy_limit_border:
                mirrored_points.append((x, 2 - y))

            # Mirror to the bottom
            if y < copy_limit_border:
                mirrored_points.append((x, -y))

            # Mirror to the top-right corner
            if x > 1-copy_limit_corner and y > 1-copy_limit_corner:
                mirrored_points.append((2 - x, 2 - y))

            # Mirror to the top-left corner
            if x < copy_limit_corner and y > 1-copy_limit_corner:
                mirrored_points.append((-x, 2 - y))

            # Mirror to the bottom-right corner
            if x > 1-copy_limit_corner and y < copy_limit_corner:
                mirrored_points.append((2 - x, -y))

            # Mirror to the bottom-left corner
            if x < copy_limit_corner and y < copy_limit_corner:
                mirrored_points.append((-x, -y))

        return np.array(mirrored_points)
    
    def compute_spearman_rho(self):
        width_square = self.width ** 2
        c_00 = np.zeros((self.m-1, self.m-1))
        c_10 = np.zeros((self.m-1, self.m-1))
        c_01 = np.zeros((self.m-1, self.m-1))
        c_11 = np.zeros((self.m-1, self.m-1))

        for i in range(self.m-1):
            for j in range(self.m-1):
                x1, x2 = i / (self.m - 1), (i + 1) / (self.m - 1)
                y1, y2 = j / (self.m - 1), (j + 1) / (self.m - 1)

                c_00[i,j] = (2*x1 + x2) * (2*y1 + y2)
                c_10[i,j] = (x1 + 2*x2) * (2*y1 + y2)
                c_01[i,j] = (2*x1 + x2) * (y1 + 2*y2)
                c_11[i,j] = (x1 + 2*x2) * (y1 + 2*y2)

        # Compute the Spearman rho coefficient
        spearman_rho = np.sum(np.multiply(self.grid[:-1,:-1],c_00) + np.multiply(self.grid[1:,:-1],c_10) + np.multiply(self.grid[:-1,1:],c_01) + np.multiply(self.grid[1:,1:],c_11)) * width_square/3 - 3
        return spearman_rho

    def show(self, mode='pdf'):
        xpos, ypos = np.meshgrid(
            np.linspace(0, 1, self.m),
            np.linspace(0, 1, self.m)
        )
        zpos = np.zeros_like(xpos)

        for i in range(self.m):
            for j in range(self.m):
                if mode == 'pdf':
                    zpos[i, j] = self.grid[i, j]
                elif mode == 'cdf':
                    zpos[i, j] = self.cdf(xpos[i, j], ypos[i, j])
                else:
                    raise ValueError("Invalid mode. Expected 'pdf' or 'cdf'.")

        # Use Plotly for 3D plotting
        fig = go.Figure(data=[go.Surface(z=zpos, x=xpos, y=ypos)])
        fig.update_layout(title='3D Plot of bivariate Copula', autosize=True,
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
        cell_integrals[:x0,:y0]=self.cell_volume[:x0,:y0]
        

        for i in range(x0):
            cell_integrals[i,y0]=((self.grid[i,y0]*(2-dy)+self.grid[i,y1]*dy)+(self.grid[i+1,y0]*(2-dy)+self.grid[i+1,y1]*dy))*dy*self.width**2/4

        for j in range(y0):
            cell_integrals[x0,j]=((self.grid[x0,j]*(2-dx)+self.grid[x1,j]*dx)+(self.grid[x0,j+1]*(2-dx)+self.grid[x1,j+1]*dx))*dx*self.width**2/4

        cell_integrals[x0,y0]=((self.grid[y0, x0] * (1 - dx) * (1 - dy) +
                 self.grid[y1, x0] * (1 - dx) * dy +
                 self.grid[y0, x1] * dx * (1 - dy) +
                 self.grid[y1, x1] * dx * dy)+self.grid[y0, x0]+(self.grid[x0,y0]*(1-dx)+self.grid[x1,y0]*dx)+(self.grid[x0,y0]*(1-dy)+self.grid[x0,y1]*dy))*dx*dy*self.width**2/4
        
        return np.sum(cell_integrals)

    def bilinear_interpolation(self, x, y):
        x_index = x * (self.m - 1)
        y_index = y * (self.m - 1)
        x0, y0 = int(x_index), int(y_index)
        x1, y1 = min(x0 + 1, self.m - 1), min(y0 + 1, self.m - 1)
        dx, dy = x_index - x0, y_index - y0
        # Interpolate
        value = (self.grid[y0, x0] * (1 - dx) * (1 - dy) +
                 self.grid[y1, x0] * (1 - dx) * dy +
                 self.grid[y0, x1] * dx * (1 - dy) +
                 self.grid[y1, x1] * dx * dy)
        return value
    
    def pdf(self, points):
        x_indices = points[:, 0] * (self.m - 1)
        y_indices = points[:, 1] * (self.m - 1)
        x0, y0 = np.floor(x_indices).astype(int), np.floor(y_indices).astype(int)
        x1, y1 = np.minimum(x0 + 1, self.m - 1), np.minimum(y0 + 1, self.m - 1)
        dx, dy = x_indices - x0, y_indices - y0
        # Interpolate
        values = (self.grid[y0, x0] * (1 - dx) * (1 - dy) +
                self.grid[y1, x0] * (1 - dx) * dy +
                self.grid[y0, x1] * dx * (1 - dy) +
                self.grid[y1, x1] * dx * dy)
        return values
    
    def entropy(self):
        # Define the entropy function for NumPy arrays
        def np_entropy(x):
            masked_x = np.ma.masked_where(x == 0, x)
            result = -masked_x * np.ma.log(masked_x)
            return result.filled(0)
        # Calculate entropy for corners, edges, and center
        f_corners = np_entropy(self.grid[0:-1,0:-1]) + np_entropy(self.grid[1:,0:-1]) + np_entropy(self.grid[0:-1,1:]) + np_entropy(self.grid[1:,1:])
        f_edges = 4*(np_entropy((self.grid[0:-1,0:-1]+self.grid[1:,0:-1])/2) + np_entropy((self.grid[0:-1,1:]+self.grid[1:,1:])/2) + np_entropy((self.grid[0:-1,0:-1]+self.grid[0:-1,1:])/2) + np_entropy((self.grid[1:,0:-1]+self.grid[1:,1:])/2))
        f_center = 16*np_entropy((self.grid[0:-1,0:-1]+self.grid[1:,0:-1]+self.grid[0:-1,1:]+self.grid[1:,1:])/4)

        # Combine the entropy expressions with Simpson's rule weights and scaling
        simpson_approx = (f_corners + f_edges + f_center) * (self.width**2/36)

        # Since we're summing over all elements, sum the resulting matrix for the total approximation
        return np.sum(simpson_approx)
    def __call__(self, x, y):
        return self.bilinear_interpolation(x, y)

class GridCopulaCopula(GridCopula):
    def __init__(self, model, m):
        super().__init__(m)
        self.grid = self.optimize_grid(self.generate_copula_points(model))

    def generate_copula_points(self, model):
        epsilon = 1e-5/self.m  # Small offset from 0 and 1
        x = np.linspace(0, 1, self.m)
        y = np.linspace(0, 1, self.m)
        x[0]=epsilon
        x[self.m-1]=1-epsilon
        y[0]=epsilon
        y[self.m-1]=1-epsilon
        X, Y = np.meshgrid(x, y)

        return model.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape((self.m, self.m))
    

    def optimize_grid(self, input_grid):
        # Create a variable grid of the same size as the input grid
        x = cp.Variable((self.m, self.m), nonneg=True)

        # Objective: Minimize the sum of squared differences from the input grid
        objective = cp.Minimize(cp.sum_squares(x - input_grid))

        # Constraints
        constraints = [
            cp.sum(self.width / 2 * (x[:-1,:] + x[1:,:]), axis=0) == np.ones((self.m)),  # column sums
            cp.sum(self.width / 2 * (x[:,:-1] + x[:,1:]), axis=1) == np.ones((self.m)),     # row sums
        ]

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        return x.value
    
class GridCopulaData(GridCopula):

    def __init__(self, data_points, m):
        super().__init__(m)
        self.std_dev = 1.3*self.width
        self.grid = self.optimize_grid(self.normalize_volume(self.count_points_with_gaussian(data_points)))

    def gaussian_2d(self, x, y, mu_x, mu_y, sigma):
        return np.exp(-(((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)))

    def count_points_with_gaussian(self, data_points):
        data_points = self.mirror_data(data_points,self.std_dev)
        grid = np.zeros((self.m, self.m))
        cell_size = 1.0 / (self.m - 1)
        for i in range(self.m):
            for j in range(self.m):
                x_val = i * cell_size
                y_val = j * cell_size
                grid[i, j] = sum(self.gaussian_2d(x, y, x_val, y_val, self.std_dev) for x, y in data_points)
        return grid

    def optimize_grid(self, input_grid):

        half_width = self.width / 2
        # Create a variable grid of the same size as the input grid
        x = cp.Variable((self.m, self.m))

        # Objective: Minimize the sum of squared differences from the input grid
        objective = cp.Minimize(cp.sum_squares(x - input_grid))

        # Constraints
        constraints = [
            cp.sum(half_width * (x[:-1,:] + x[1:,:]), axis=0) == np.ones((self.m)),  # column sums
            cp.sum(half_width * (x[:,:-1] + x[:,1:]), axis=1) == np.ones((self.m)),     # row sums
            x >= 0                                                                                        # non-negativity
        ]

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        return x.value
    
class GridCopulaEntropy(GridCopula):
    def __init__(self, spearman_rho, m, simpson=True):
        super().__init__(m)
        self.grid=self.optimize_grid(spearman_rho, simpson)
        self.cell_volume = self.cell_volume_compute()

    def optimize_grid(self, spearman_rho, simpson):
        width_square = self.width ** 2
        half_width = self.width / 2
        
        # Define the objective
        x = cp.Variable((self.m, self.m), nonneg=True)
        
        #define objective before assigning it
        entropy_approximation = 0
        if simpson:
            # Expressions for Simpson's rule approximation over the entire grid
            f_corners = cp.entr(x[0:-1,0:-1]) + cp.entr(x[1:,0:-1]) + cp.entr(x[0:-1,1:]) + cp.entr(x[1:,1:])
            f_edges = 4*(cp.entr((x[0:-1,0:-1]+x[1:,0:-1])/2) + cp.entr((x[0:-1,1:]+x[1:,1:])/2) + cp.entr((x[0:-1,0:-1]+x[0:-1,1:])/2) + cp.entr((x[1:,0:-1]+x[1:,1:])/2))
            f_center = 16*cp.entr((x[0:-1,0:-1]+x[1:,0:-1]+x[0:-1,1:]+x[1:,1:])/4)
            # Combine the entropy expressions with Simpson's rule weights and scaling
            entropy_approximation = (f_corners + f_edges + f_center) * (width_square/36)
        else:
            # Define the weights matrix w
            w = np.ones((self.m, self.m))
            w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 0.5
            w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 0.25
            w/=w.sum()
            entropy_approximation = cp.multiply(cp.entr(x), w)

        objective = cp.Maximize(cp.sum(entropy_approximation))

        # Define constraints
        c = np.zeros((self.m-1,self.m-1,4))
        for i in range(self.m-1):
            for j in range(self.m-1):
                a = i * self.width
                b = j * self.width
                c[i,j,0] = 1/3 * ((width_square)) * (3*a + self.width)   * (3*b + self.width)
                c[i,j,1] = 1/3 * ((width_square)) * (3*a + 2*self.width) * (3*b + self.width)
                c[i,j,2] = 1/3 * ((width_square)) * (3*a + self.width)   * (3*b + 2*self.width)
                c[i,j,3] = 1/3 * ((width_square)) * (3*a + 2*self.width) * (3*b + 2*self.width)

        constraints = [
            cp.sum((x[:-1,:] + x[1:,:]), axis=0) == np.full((self.m),1/half_width),  # column sums
            cp.sum((x[:,:-1] + x[:,1:]), axis=1) == np.full((self.m),1/half_width),  # row sums
            cp.sum(cp.multiply(x[:-1,:-1],c[:,:,0]) + cp.multiply(x[1:,:-1],c[:,:,1]) + cp.multiply(x[:-1,1:],c[:,:,2]) + cp.multiply(x[1:,1:],c[:,:,3])) - 3 == spearman_rho,
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        return x.value


class GridCopulaLogLikelihood(GridCopula):
    def __init__(self, data, m):
        super().__init__(m)
        self.grid=self.optimize_grid(data)
        self.cell_volume = self.cell_volume_compute()

    def optimize_grid(self, data):
        x = cp.Variable((self.m, self.m), nonneg=True)

        index_x=np.zeros((len(data),2),dtype=int)
        index_y=np.zeros((len(data),2),dtype=int)
        deltas=np.zeros((len(data),4))

        for ind, d in enumerate(data):
            x_index = d[0] * (self.m - 1)
            y_index = d[1] * (self.m - 1)
            
            index_x[ind,0]=int(x_index)
            index_x[ind,1]=min(index_x[ind,0] + 1, self.m - 1)
            
            index_y[ind,0]=int(y_index)
            index_y[ind,1]=min(index_y[ind,0] + 1, self.m - 1)

            dx=x_index - int(x_index)
            dy=y_index - int(y_index)
            deltas[ind,0]=(1-dx)*(1-dy)
            deltas[ind,1]=(1-dx)*dy
            deltas[ind,2]=dx*(1-dy)
            deltas[ind,3]=dx*dy
        
        # Interpolate
        log_likelihood = cp.sum(cp.log(cp.multiply(x[index_x[:,0], index_y[:,0]],deltas[:,0]) +
                 cp.multiply(x[index_x[:,0], index_y[:,1]], deltas[:,1]) +
                 cp.multiply(x[index_x[:,1], index_y[:,0]], deltas[:,2]) +
                 cp.multiply(x[index_x[:,1], index_y[:,1]], deltas[:,3])))

        # Compute second-order differences
        diff_xx = x[:-2, :] - 2 * x[1:-1, :] + x[2:, :]
        diff_yy = x[:, :-2] - 2 * x[:, 1:-1] + x[:, 2:]
        diff_xy = x[:-2, :-2] - 2 * x[1:-1, 1:-1] + x[2:, 2:]
        diff_yx = x[2:, :-2] - 2 * x[1:-1, 1:-1] + x[:-2, 2:]

        # Compute the sum of squared second-order differences
        smoothness_penalty = cp.sum_squares(diff_xx) + cp.sum_squares(diff_yy) + (cp.sum_squares(diff_xy) + cp.sum_squares(diff_yx))/np.sqrt(2)

        lambda_ = 0.0
        objective = cp.Maximize(log_likelihood - lambda_ * smoothness_penalty)

        constraints = [
            cp.sum((x[:-1,:] + x[1:,:]), axis=0) == np.full((self.m),2/self.width),  # column sums
            cp.sum((x[:,:-1] + x[:,1:]), axis=1) == np.full((self.m),2/self.width),  # row sums
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, max_iter=500)
        print(problem.value)
        return x.value
    
class GridCopulaLogLikelihoodSpearman(GridCopula):
    def __init__(self, data, m):
        super().__init__(m)
        self.spearman_rho=spearmanr(data)[0]
        self.grid=self.optimize_grid(data)
        self.cell_volume = self.cell_volume_compute()

    def optimize_grid(self, data):
        x = cp.Variable((self.m, self.m), nonneg=True)

        index_x=np.zeros((len(data),2),dtype=int)
        index_y=np.zeros((len(data),2),dtype=int)
        deltas=np.zeros((len(data),4))

        for ind, d in enumerate(data):
            x_index = d[0] * (self.m - 1)
            y_index = d[1] * (self.m - 1)
            
            index_x[ind,0]=int(x_index)
            index_x[ind,1]=min(index_x[ind,0] + 1, self.m - 1)
            
            index_y[ind,0]=int(y_index)
            index_y[ind,1]=min(index_y[ind,0] + 1, self.m - 1)

            dx=x_index - int(x_index)
            dy=y_index - int(y_index)
            deltas[ind,0]=(1-dx)*(1-dy)
            deltas[ind,1]=(1-dx)*dy
            deltas[ind,2]=dx*(1-dy)
            deltas[ind,3]=dx*dy
        
        # Interpolate
        log_likelihood = cp.sum(cp.log(cp.multiply(x[index_x[:,0], index_y[:,0]],deltas[:,0]) +
                 cp.multiply(x[index_x[:,0], index_y[:,1]], deltas[:,1]) +
                 cp.multiply(x[index_x[:,1], index_y[:,0]], deltas[:,2]) +
                 cp.multiply(x[index_x[:,1], index_y[:,1]], deltas[:,3])))

        # Compute second-order differences
        diff_xx = x[1:-3, 1:-1] - 2 * x[2:-2, 1:-1] + x[3:-1, 1:-1]
        diff_yy = x[1:-1, 1:-3] - 2 * x[1:-1, 2:-2] + x[1:-1, 3:-1]
        diff_xy = x[1:-3, 1:-3] - 2 * x[2:-2, 2:-2] + x[3:-1, 3:-1]
        diff_yx = x[3:-1, 1:-3] - 2 * x[2:-2, 2:-2] + x[1:-3, 3:-1]

        # Compute the sum of squared second-order differences
        smoothness_penalty = cp.sum_squares(diff_xx) + cp.sum_squares(diff_yy) + (cp.sum_squares(diff_xy) + cp.sum_squares(diff_yx))/np.sqrt(2)

        lambda_ = 0.4
        objective = cp.Maximize(log_likelihood - lambda_ * smoothness_penalty)

        # Define constraints
        c = np.zeros((self.m-1,self.m-1,4))
        for i in range(self.m-1):
            for j in range(self.m-1):
                a = i * self.width
                b = j * self.width
                c[i,j,0] = 1/3 * (self.width**2) * (3*a + self.width)   * (3*b + self.width)
                c[i,j,1] = 1/3 * (self.width**2) * (3*a + 2*self.width) * (3*b + self.width)
                c[i,j,2] = 1/3 * (self.width**2) * (3*a + self.width)   * (3*b + 2*self.width)
                c[i,j,3] = 1/3 * (self.width**2) * (3*a + 2*self.width) * (3*b + 2*self.width)


        constraints = [
            cp.sum((x[:-1,:] + x[1:,:]), axis=0) == np.full((self.m),2/self.width),  # column sums
            cp.sum((x[:,:-1] + x[:,1:]), axis=1) == np.full((self.m),2/self.width),  # row sums
            cp.sum(cp.multiply(x[:-1,:-1],c[:,:,0]) + cp.multiply(x[1:,:-1],c[:,:,1]) + cp.multiply(x[:-1,1:],c[:,:,2]) + cp.multiply(x[1:,1:],c[:,:,3])) - 3 == self.spearman_rho
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, max_iter=500)
        return x.value
