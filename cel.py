import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

#Ellipse Parameter
a = 10  # Semi-major axis
e = 0.5  # Eccentricity
n_points=50 #number of points

#Ellipse in Polar Coords, Sun at Center(0,0)
theta = np.linspace(0, 2 * np.pi, n_points) #True anomaly (just linear, "speed" wrong)
r = a * (1 - e**2) / (1 + e * np.cos(theta))

#Ellipse in Cartesian Coords, Sun at Center(0,0), no inclination = in xy-Plane)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = 0


#Plot Ellipse
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label="Ellipse")

#Plot Animation
def update(frame):
    # Update position
    x_pos = r[frame] * np.cos(theta[frame])
    y_pos = r[frame] * np.sin(theta[frame])
    z_pos = 0 
    # Create points
    ax.scatter(x_pos, y_pos, z_pos , color='red', s=20)

ani = FuncAnimation(fig, update, frames=n_points, interval=50, repeat=True)

plt.show()





