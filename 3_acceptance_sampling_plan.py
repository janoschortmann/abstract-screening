#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:27:42 2024

@author: isabel
"""

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.interactive(True)


# Define the file path
file_path = "config3.txt"

# Initialize a dictionary to store the variables
variables = {}


with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        if line:

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')  

            variables[key] = value

# Access the variables
alpha = float(variables.get("alpha"))
beta = float(variables.get("beta"))
p1 = float(variables.get("p1"))
p2 = float(variables.get("p2"))

# Print the values
print("The configuration file has been read.")
print("alpha:", alpha)
print("beta:", beta)
print("p1:", p1)
print("p2:", p2)


# Function to find the appropriate (n, c)
def find_nc(alpha, beta, p1, p2):
    for n in range(1, 1000):  # Searching for n
        for c in range(n):  # Searching for c
            # Calculate producer's risk
            producer_risk = sum([binom.pmf(k, n, p1) for k in range(c + 1)])
            # Calculate consumer's risk
            consumer_risk = sum([binom.pmf(k, n, p2) for k in range(c + 1)])
            
            if producer_risk >= 1 - alpha and consumer_risk <= beta:
                return n, c

# Find n and c
n, c = find_nc(alpha, beta, p1, p2)
print(f"\nSample size (n): {n}, Acceptance number (c): {c}")

# Function to calculate the probability of acceptance
def probability_of_acceptance(n, c, p):
    return sum([binom.pmf(k, n, p) for k in range(c + 1)])

# Plotting the OC curve
p_values = np.linspace(0, 0.5, 500)

acceptance_probabilities = [probability_of_acceptance(n, c, p) for p in p_values]



plt.plot(p_values, acceptance_probabilities, label=f'n = {n}, c = {c}')
plt.axhline(1 - alpha, color='red', linestyle='--', label=f'1 - alpha = {1 - alpha}')
plt.axhline(beta, color='blue', linestyle='--', label=f'beta = {beta}')
plt.axvline(p1, color='green', linestyle='--', label=f'p1 = {p1}')
plt.axvline(p2, color='purple', linestyle='--', label=f'p2 = {p2}')
plt.xlabel('Fraction Defective (p)')
plt.ylabel('Probability of Acceptance')
plt.title('Operating Characteristic (OC) Curve')
plt.legend()
plt.grid(True)
plt.show()