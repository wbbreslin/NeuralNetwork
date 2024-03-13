import matplotlib.pyplot as plt
import Highams2019_Hessian_Pytorch as torch_module
import Highams2019_Hessian_SOA as soa_module

SOA = soa_module.full_hessian[:, 0]
torch_array = torch_module.hessian_matrix[:, 0].numpy()
dif = SOA - torch_array

# Plotting SOA and Torch on the first graph with different markers
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(SOA, marker='o', linestyle='-', label='SOA')  # Marker 'o' for SOA
plt.plot(torch_array, marker='x', linestyle='--', label='Torch')  # Marker 'x' for Torch
plt.title('SOA vs Torch')
plt.legend()

# Plotting the difference on the second graph
plt.subplot(1, 2, 2)
plt.plot(dif, color='red')
plt.title('Difference (SOA - Torch)')

plt.tight_layout()
plt.show()
