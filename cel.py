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
peri_offset= 70 #degree around z

n_points= 50 #number of points

#Ellipse in Polar Coords, "Sun" at Center(0,0)
theta = np.linspace(0, 2 * np.pi, n_points) #True anomaly (just linear, "speed" wrong)
r = (a * (1 - e**2)) / (1 + e * np.cos(theta))

#Ellipse in Cartesian Coords, Sun at Center(0,0), no inclination = in xy-Plane)
own_sys_x = r * np.cos(theta)
own_sys_y = r * np.sin(theta)
own_sys_z = np.zeros_like(theta)
own_sys_vec = np.vstack((own_sys_x,own_sys_y,own_sys_z))
print(type(own_sys_vec))

#Input Transformation
rad_incline=np.deg2rad(incline)
rad_z_incl=np.deg2rad(z_dir_incline)


def rotate_around_z(vec, deg):
    if isinstance(vec, np.ndarray):
        print("good input")
        rad=np.deg2rad(deg)
        z_rot_mat=np.array([   # Rotation around z-Axis
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1] 
        ])
        z_rot_vec= np.dot(z_rot_mat, vec)
        return z_rot_vec
    else:
        print("nah bro, z-rot imput is not an array")
        return 
    

def rotate_around_x(vec,deg):
    if isinstance(vec, np.ndarray):
        print("good input")
        rad=np.deg2rad(deg)
        x_rot_mat=np.array([   #Inclination
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]  
        ])
        x_rot_vec = np.dot(x_rot_mat, vec)
        return x_rot_vec
    else:
        print("nah bro, x-rot imput is not an array")
        return 

def all_orbit_adjustments(vec, peri_offset, incl, incl_rot):
    vec=rotate_around_z(vec,peri_offset)
    vec=rotate_around_x(vec, incl)
    vec=rotate_around_z(vec, incl_rot)
    return vec




#Rotational Matrix for Inclinations
rot_mat_incl=np.array([   #Inclination
    [1, 0, 0],
    [0, np.cos(rad_incline), -np.sin(rad_incline)],
    [0, np.sin(rad_incline), np.cos(rad_incline)]  
])

z_rot_mat_incl=np.array([   # Rotation around z-Axis
    [np.cos(rad_z_incl), -np.sin(rad_z_incl), 0],
    [np.sin(rad_z_incl), np.cos(rad_z_incl), 0],
    [0, 0, 1]  
])

cart_sys_w_incline =all_orbit_adjustments(own_sys_vec, peri_offset, incline, z_dir_incline)


#Plot Ellipse
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(cart_sys_w_incline[0],cart_sys_w_incline[1],cart_sys_w_incline[2], label="Ellipse")
ax.scatter(0, 0, 0 , color='yellow', s=30)
ax.set_aspect('equal')
ax.plot()



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