import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = """
y 0.0001633384672459215
z 0.0003058532893192023
aa 0.0001323176984442398
ab 0.0002861683606170118
ac 0.000581087835598737
ad 0.0029117814265191555
ae 0.0002099314151564613
af 0.0002885438734665513
ag 0.00010162567195948213
ah 0.00021081046725157648
ai 0.00020060596580151469
aj 0.0023901937529444695
"""

data2 = """
a 0.00031753492658026516
a 0.0003254242183174938
a 0.00013473669241648167
a 0.00028711179038509727
a 0.0006688571302220225
a 0.004512597341090441
a 0.0002114680828526616
a 0.00028638666844926775
a 0.00009597464668331668
a 0.00021373599884100258
a 0.00020171825599391013
a 0.0026757384184747934
"""

data_list = [line.split() for line in data.strip().split('\n')]
df = pd.DataFrame(data_list, columns=['Sweep', 'Loss'])
df['Loss'] = df['Loss'].astype(float)

mean_loss = df['Loss'].mean()
median_loss = df['Loss'].median()
min_loss = df['Loss'].min()
max_loss = df['Loss'].max()
std_loss = df['Loss'].std()
quantiles = df['Loss'].quantile([0.25, 0.5, 0.75])

data_list2 = [line.split() for line in data2.strip().split('\n')]
df2 = pd.DataFrame(data_list2, columns=['Sweep', 'Loss'])
df2['Loss'] = df2['Loss'].astype(float)

mean_loss2 = df2['Loss'].mean()
median_loss2 = df2['Loss'].median()
min_loss2 = df2['Loss'].min()
max_loss2 = df2['Loss'].max()
std_loss2 = df2['Loss'].std()
quantiles2 = df2['Loss'].quantile([0.25, 0.5, 0.75])

print("L1 Regularization vs. OPR")
print(f"Mean Loss: {mean_loss}, {mean_loss2}")
print(f"Median Loss: {median_loss}, {median_loss2}")
print(f"Minimum Loss: {min_loss}, {min_loss2}")
print(f"Maximum Loss: {max_loss}, {max_loss2}")
print(f"Standard Deviation: {std_loss}, {std_loss2}")
print(f"25th Percentile: {quantiles[0.25]}, {quantiles2[0.25]}")
print(f"50th Percentile (Median): {quantiles[0.5]}, {quantiles[0.5]}")
print(f"75th Percentile: {quantiles[0.75]}, {quantiles2[0.75]}")

top_5 = df.nsmallest(5, 'Loss')
bottom_5 = df.nlargest(5, 'Loss')
top_52 = df2.nsmallest(5, 'Loss')
bottom_52 = df2.nlargest(5, 'Loss')
print("Lowest L1 Loss:")
print(top_5)
print("Highest L1 Loss:")
print(bottom_5)
print("Lowest OPR Loss:")
print(top_52)
print("Highest OPR Loss:")
print(bottom_52)

plt.figure(figsize=(8, 5))
plt.plot(df['Loss'].sort_values(ascending=False).reset_index(drop=True), label='Sorted Validation Values', marker='o')
plt.axhline(mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.2e}')
plt.axhline(median_loss, color='g', linestyle='--', label=f'Median: {median_loss:.2e}')
plt.axhline(min_loss, color='b', linestyle='--', label=f'Min: {min_loss:.2e}')
plt.axhline(max_loss, color='purple', linestyle='--', label=f'Max: {max_loss:.2e}')
plt.xlabel('Sorted Sweeps')
plt.ylabel('Validation Value')
plt.title('L1 Validation Metrics')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df2['Loss'].sort_values(ascending=False).reset_index(drop=True), label='Sorted Validation Values', marker='o')
plt.axhline(mean_loss2, color='r', linestyle='--', label=f'Mean: {mean_loss2:.2e}')
plt.axhline(median_loss2, color='g', linestyle='--', label=f'Median: {median_loss2:.2e}')
plt.axhline(min_loss2, color='b', linestyle='--', label=f'Min: {min_loss2:.2e}')
plt.axhline(max_loss2, color='purple', linestyle='--', label=f'Max: {max_loss2:.2e}')
plt.xlabel('Sorted Sweeps')
plt.ylabel('Validation Value')
plt.title('OPR Validation Metrics')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()