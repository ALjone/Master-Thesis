from baseline_random import baseline_random
from baseline_sklearn import baseline_sklearn
import warnings
warnings.filterwarnings("ignore")


n = 500
print("\n\n")

#WITH TIME:
T = 100
max_length = None
print(f"Baseline random with time: {T}, n = {n}:")
baseline_random(T, n, max_length)
print(f"\nBaseline sklearn with time: {T}, n = {n}:")
baseline_sklearn(T, n, max_length)

#WITH A SET AMOUNT OF ITERATIONS
T = 10000000000
max_length = 20
print(f"\nBaseline random with max_length: {max_length}, n = {n}:")
baseline_random(T, n, max_length)
print(f"\nBaseline sklearn with max_length: {max_length}, n = {n}:")
baseline_sklearn(T, n, max_length)


