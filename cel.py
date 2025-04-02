import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

#Ellipse Parameter
a = 5  # Semi-major axis
e = 0.9  # Eccentricity
b = a*np.sqrt(1-e**2) #Semi-minor axis
incline = 40 #degree compared to xy plane
z_dir_incline = 30 #degree around z- Axis from x-Axis 

n_points= 50 #number of points

#Ellipse in Polar Coords, "Sun" at Center(0,0)
theta = np.linspace(0, 2 * np.pi, n_points) #True anomaly (just linear, "speed" wrong)
r = (a * (1 - e**2)) / (1 + e * np.cos(theta))

#Ellipse in Cartesian Coords, Sun at Center(0,0), no inclination = in xy-Plane)
own_sys_x = r * np.cos(theta)
own_sys_y = r * np.sin(theta)
own_sys_z = np.zeros_like(theta)
own_sys_vec = np.vstack((own_sys_x,own_sys_y,own_sys_z))



#Input Transformation
rad_incline=np.deg2rad(incline)

#Rotational Matrix for Inclination
rot_mat_incl=np.array([
    [1, 0, 0],
    [0, np.cos(rad_incline), -np.sin(rad_incline)],
    [0, np.sin(rad_incline), np.cos(rad_incline)]  
])

cart_sys_w_incline = np.dot(rot_mat_incl, own_sys_vec)



#Plot Ellipse
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(cart_sys_w_incline[0],cart_sys_w_incline[1],cart_sys_w_incline[2], label="Ellipse")
ax.set_aspect('equal')


#Plot Animation
def update(frame):
    # Update position
    x_pos = cart_sys_w_incline[0][frame]
    y_pos = cart_sys_w_incline[1][frame]
    z_pos = cart_sys_w_incline[2][frame] 
    # Create points
    ax.scatter(x_pos, y_pos, z_pos , color='red', s=20)

ani = FuncAnimation(fig, update, frames=n_points, interval=50, repeat=True)

plt.show()