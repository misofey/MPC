import matplotlib.pyplot as plt
import numpy as np

with open("/home/miso/Downloads/Niek_RDW_skidpad_7.3ms_2024_11_23T145500_export/t0.csv") as data_file:
    data_string1 = data_file.readlines()
with open("/home/miso/Downloads/Niek_RDW_skidpad_7.3ms_2024_11_23T145500_export/t1.csv") as data_file:
    data_string2 = data_file.readlines()

# with open("//home/miso/Downloads/Niek_RDW_skidpad_5.8ms_2024_11_23T145000_export/t0.csv") as data_file:
#     data_string1 = data_file.readlines()
# with open("/home/miso/Downloads/Niek_RDW_skidpad_5.8ms_2024_11_23T145000_export/t1.csv") as data_file:
#     data_string2 = data_file.readlines()

gyro_raw = np.genfromtxt(data_string1, delimiter=",")
gss_raw = np.genfromtxt(data_string2, delimiter=",")

time = gss_raw[1:, 0]
velx = gss_raw[1:, 2]
vely = gss_raw[1:, 1]
gyro_sampled = np.interp(time, gyro_raw[:, 0], gyro_raw[:, 1])

ns = gyro_sampled.shape[0]
coeffs = np.ones((ns, 2))

coeffs[:, 0] = velx * gyro_sampled
print(np.sum(np.isnan(gyro_sampled)))

values = np.linalg.lstsq(coeffs, vely)

values[0][1] -= 0.15
print(values[0])
plt.scatter(velx * gyro_sampled, vely, c=time, s=1)
plt.plot(velx * gyro_sampled, values[0][0] * velx * gyro_sampled + values[0][1])
plt.show()
