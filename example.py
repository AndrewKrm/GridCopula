#Usage examples of the gridcopula package

import pyvinecopulib as pv
import numpy as np
import gridcopula as gc
import gridcopula2D as gc2d
import matplotlib.pyplot as plt
from scipy.stats import norm


def IAE(model, cop, data_points_test):
    return np.sum(
        np.abs(model.pdf(data_points_test) - cop.pdf(data_points_test))
        / cop.pdf(data_points_test)
    ) / len(data_points_test)


m = 20  # Resolution
rho = 0.7
samples = 2000
cop = pv.Bicop(family=pv.BicopFamily.student, parameters=[0.5, 3])

IAE_val_likelihood = []
IAE_val_copula = []
IAE_val_tll = []
model_logLikelihood = None
model_copula = None
for i in range(1, 20):
    data_points_train = cop.simulate(n=samples, seeds=[i, 2, 2])
    data_points_test = cop.simulate(n=samples, seeds=[i + 100, 4, 5])

    # model_logLikelihood=gc2d.GridCopulaLogLikelihood(data_points_train,m)
    model_copula = gc2d.GridCopulaLogLikelihoodSpearman(data_points_train, m)
    model_tll = pv.Bicop(
        data=data_points_train,
        controls=pv.FitControlsBicop(family_set=[pv.BicopFamily.tll]),
    )
    # IAE_val_likelihood.append(IAE(model_logLikelihood,cop,data_points_test))
    IAE_val_copula.append(IAE(model_copula, cop, data_points_test))
    IAE_val_tll.append(IAE(model_tll, cop, data_points_test))

# Box plot of the IAE values
plt.boxplot([IAE_val_copula, IAE_val_tll], labels=["Likelihood + spearman grid", "TLL"])
plt.title(f"Box plot of IAE values ({m}x{m} grid): gaussian, parameters=[0.7]")
plt.ylabel("IAE values")
plt.show()
model_copula.show()
model_logLikelihood.show()

"""
spearman_rho = np.array([
    [1,                 0.535294,       0.664706,       0.629412,       -0.414706],
    [0.535294,          1,              0.247059,       0.423529,       -0.4],
    [0.664706,          0.247059,       1,              0.844118,       -0.317647],
    [0.629412,          0.423529,       0.844118,       1,              -0.247059],
    [-0.414706,         -0.4,           -0.317647,      -0.247059,      1]
])
#model=gc2d.GridCopulaEntropy(0.9,m)
#model.show()
model=gc.GridCopulaEntropy(spearman_rho,m)

# Save the model to a file
model.save_to_file('model.pkl')


# Later on, you can load the model from the file
loaded_model = gc.GridCopulaEntropy.load_from_file('model.pkl')
for i in range(0,4):
    for j in range(i+1,5):
        loaded_model.set_projected_dimensions(i,j)
        loaded_model.show()
        loaded_model.show('cdf')



# Generate data_points and model
modelData = gc2d.GridCopulaData(data_points, m)
modelData.show()
"""
"""
# store the entropy of the model and the gaussian entropy and plot both lines on the same graph
import numpy as np
from scipy.stats import norm, multivariate_normal

def safe_divide(num, den):
    return np.divide(num, den, out=np.full_like(num, np.nan), where=(den != 0))


def create_grid_and_evaluate_gaussian_copula_density(m, rho):
    # Adjust linspace to avoid 0 and 1
    epsilon = 1e-5  # Small offset from 0 and 1
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, m)
    x[0]=epsilon
    x[m-1]=1-epsilon
    y[0]=epsilon
    y[m-1]=1-epsilon
    
    X, Y = np.meshgrid(x, y)
    
    X_norm = norm.ppf(X)
    Y_norm = norm.ppf(Y)
    
    covariance_matrix = [[1, rho], [rho, 1]]
    
    copula_density_values = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            joint_density = multivariate_normal.pdf([X_norm[i, j], Y_norm[i, j]], mean=[0, 0], cov=covariance_matrix)
            marginal_density_x = norm.pdf(X_norm[i, j])
            marginal_density_y = norm.pdf(Y_norm[i, j])
            
            # Use safe division
            copula_density_values[i, j] = safe_divide(joint_density, marginal_density_x * marginal_density_y)
    
    return copula_density_values

# Example usage
m = 6  # Grid size
for m in range(8, 50, 5):
    rho = 0.7
    rho_spearman = (6 / np.pi) * np.arcsin(rho / 2)
    density_values = create_grid_and_evaluate_gaussian_copula_density(m, rho)
    modelCopula=gc2d.GridCopulaCopula(density_values,m,rho_spearman)
    modelEntropy=gc2d.GridCopulaEntropy(rho_spearman,m, simpson=True)
    print("resolution (m): ",m)
    print("ModelCopula entropy: ",modelCopula.entropy())
    print("ModelEntropy entropy: ",modelEntropy.entropy())
    print("Gaussian entropy: ",1/2*np.log(1-rho**2))
    #modelEntropy.show()
    print("_________________________")

"""

"""
m=30
rho=0.7
modelEntropyVal=[]
modelEntropySimpsonVal=[]
gaussianEntropyVal=[]
m_values=np.arange(95,98,1)
modelEntropyVal=[]
modelEntropySimpsonVal=[]
gaussianEntropyVal=[]
for m in m_values:
    rho_spearman = (6 / np.pi) * np.arcsin(rho / 2)
    model=gc2d.GridCopulaEntropy(rho_spearman,m, False)
    modelSimpson=gc2d.GridCopulaEntropy(rho_spearman,m)
    modelSimpson.show()
    modelEntropyVal.append(-model.entropy())
    modelEntropySimpsonVal.append(-modelSimpson.entropy())
    gaussianEntropyVal.append(-1/2*np.log(1-(rho)**2))
    print("m: ",m)

plt.plot(m_values,modelEntropyVal, label='Model Entropy', linestyle='-')
plt.plot(m_values,modelEntropySimpsonVal, label='Model Entropy Simpson', linestyle='--')
plt.plot(m_values,gaussianEntropyVal, label='Gaussian Entropy')
plt.title("Model Entropy vs Gaussian Entropy")  # Add title here
plt.legend()  # Display the labels
plt.yscale('log')
plt.show()

# plot the difference between the two
plt.plot(m_values,np.abs(np.array(modelEntropyVal)-np.array(gaussianEntropyVal)), label='Difference normal')
plt.plot(m_values,np.abs(np.array(modelEntropySimpsonVal)-np.array(gaussianEntropyVal)), label='Difference simpson')
plt.title("Difference between Model Entropy and Gaussian Entropy")  # Add title here
plt.legend()  # Display the label
plt.yscale('log')
plt.show()
"""
"""
for rho in np.linspace(0,0.9,10):
    print("correlation: ",rho)
    modelEntropy = gc2d.GridCopulaEntropy(rho, m)
    print("Model entropy: ",-modelEntropy.entropy())
    print("Gaussian entropy: ",-1/2*np.log(1-rho**2))
    print("Difference: ",-modelEntropy.entropy()+1/2*np.log(1-rho**2))
    print("_________________________")

"""
