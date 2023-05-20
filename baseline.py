from baseline_random import baseline_random
from baseline_sklearn import baseline_sklearn
from baseline_gpy import baseline_gpy
import warnings
warnings.filterwarnings("ignore")


n = 2
print("\n\n")

#WITH TIME:
max_length = None
print(f"Baseline random with n = {n}:")
baseline_random(n, max_length)
print(f"\nBaseline gpy with n = {n}:")
baseline_gpy(n, max_length, learning_rate=0.1, training_iters=200)
print(f"\nBaseline sklearn with n = {n}:")
baseline_sklearn(n, max_length)

#WITH A SET AMOUNT OF ITERATIONS
max_length = 20
print(f"\nBaseline random with max_length: {max_length}, n = {n}:")
baseline_random(n, max_length)
print(f"\nBaseline gpy with max_length: {max_length}, n = {n}:")
baseline_gpy(n, max_length, learning_rate=0.1, training_iters=200)
print(f"\nBaseline sklearn with max_length: {max_length}, n = {n}:")
baseline_sklearn(n, max_length)