import matplotlib.pyplot as plt
import Highams2019_Hessian_FDM as fdm_module
import Highams2019_Hessian_SOA as soa_module

SOA = soa_module.full_hessian[:, 0].reshape(-1,1)
FDM = fdm_module.full_hessian[:, 0]
dif = SOA - FDM

# Plotting SOA and Torch on the first graph with different markers
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(SOA, marker='o', linestyle='-', label='SOA')  # Marker 'o' for SOA
plt.plot(FDM, marker='x', linestyle='--', label='FDM')  # Marker 'x' for Torch
plt.title('SOA vs Torch')
plt.legend()

# Plotting the difference on the second graph
plt.subplot(1, 2, 2)
plt.plot(dif, color='red')
plt.title('Difference (SOA - FDM)')

plt.tight_layout()
plt.show()
