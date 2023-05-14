import numpy as np
from sklearn.decomposition import FactorAnalysis


# Function to generate the FA dataset
def generate_fa_dataset(N, n, m, sigma_square, A):
    y = np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m))
    e = np.random.multivariate_normal(mean=np.zeros(n), cov=sigma_square * np.eye(n))
    x = np.dot(A, y) + e
    return x


# Parameters
N = 100  # Sample size
n = 10  # Dimensionality of observed variable x
m_values = [1, 2, 3, 4, 5]  # Different values for dimensionality of latent variable y
sigma_square = 0.1
A = np.random.randn(n, max(m_values))  # Randomly initialize the loading matrix A

# Generate the datasets
datasets = []
for _ in range(N):
    datasets.append(generate_fa_dataset(N, n, max(m_values), sigma_square, A))

# Model selection using BIC and AIC
results_aic = []
results_bic = []

for m in m_values:
    log_likelihoods = []
    for dataset in datasets:
        dataset = dataset.reshape(-1, 1)  # Reshape the dataset to a column vector
        fa = FactorAnalysis(n_components=m)
        fa.fit(dataset)
        log_likelihood = fa.score(dataset)  # Calculate log-likelihood using the score method
        log_likelihoods.append(log_likelihood)

    num_free_params = m * n + m * (m + 1) / 2  # Number of free parameters in the FA model

    aic = np.array(log_likelihoods) - num_free_params
    bic = np.array(log_likelihoods) - (np.log(N) / 2) * num_free_params

    results_aic.append(np.mean(aic))
    results_bic.append(np.mean(bic))

# Print the results
print("AIC results:")
for m, aic in zip(m_values, results_aic):
    print(f"m = {m}: AIC = {aic}")

print("\nBIC results:")
for m, bic in zip(m_values, results_bic):
    print(f"m = {m}: BIC = {bic}")
