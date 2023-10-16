import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
# Global Test
def global_test(data):
    observed_freq, _ = np.histogram(data, bins='auto')
    expected_freq = np.mean(data) * len(data)
    chi2_statistic = np.sum((observed_freq - expected_freq)**2 / expected_freq)
    p_value = 1 - chi2.cdf(chi2_statistic, df=len(data)-1)
    return p_value

# Local Test
def local_test(data, threshold):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    p_values = 2 * (1 - norm.cdf(abs(z_scores)))
    for i, p_value in enumerate(p_values):
        if p_value < threshold:
            outliers.append(i)
    return outliers

# Example data
data = [10.2, 9.5, 8.7, 11.1, 10.8, 9.9, 12.5, 10.4, 8.9, 11.3, 100.0, 9.6, 10.1]

# Set the significance level (alpha)
alpha = 0.05

# Perform global test
global_p_value = global_test(data)
critical_value = chi2.ppf(1 - alpha, df=len(data)-1)
print("Global Test P-value:", global_p_value)
print("Critical Value:", critical_value)

# Compare the p-value with the critical value
if global_p_value < alpha:
    print("Reject the null hypothesis. There is evidence of a global effect.")
else:
    print("Fail to reject the null hypothesis. There is no evidence of a global effect.")

# Perform local test
local_outliers = local_test(data, threshold=alpha)
for i in range(len(local_outliers)):
	print("Local Test Outliers",data[local_outliers[i]])
