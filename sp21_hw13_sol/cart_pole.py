import arcade
import numpy as np
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import time

# Constants for drawing
FPS = 60
CART_HEIGHT = 100
CART_WIDTH = 200
WHEEL_RAD = 20
BALL_RAD = 15
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
GROUND_HEIGHT = 200
SCREEN_TITLE = "Cart-Pole"

# System constants
m = 14
M = 20
g = 0.5
l = 250
k = 1

# Linearized system
A = np.array([
	[0, 1, 0, 0],
	[0, -k/M, -(m*g)/M, 0],
	[0, 0, 0, 1],
	[0, k/(M*l), ((M+m)*g)/(M*l), 0]
])
b = np.array([[0], [1/M], [0], [-1/(M*l)]])

# Define variables for the Cart-Pole System
states = np.array([[0,0,0.1,0]])
u = 0
count = FPS * 2

def calc_inputs(A, b):
	Delta = 1
	# Eigendecomposition of A
	eigs, V = np.linalg.eig(A)
	V_inv = np.linalg.inv(V)

	# Convert to discrete time parameters A_d and \vec{b}_d
	Lambda_d = np.diag(np.exp(eigs * Delta))
	A_d = V @ Lambda_d @ V_inv

	num = np.exp(eigs * Delta) - 1
	diag = np.ones_like(eigs) * Delta
	for idx, ele in enumerate(eigs):
	    if ele:
	        diag[idx] = num[idx] / ele
	M_d = np.diag(diag)
	b_d = V @ M_d @ V_inv @ b

	Ab = A_d @ b_d
	A2b = A_d @ Ab
	A3b = A_d @ A2b
	C = np.hstack([b_d, Ab, A2b, A3b])

	state_final = np.array([[0], [0], [0], [0]])
	state_initial = np.array([[x], [x_dot], [theta], [theta_dot]])
	A4 = A_d @ A_d @ A_d @ A_d
	u_d = np.linalg.solve(C, state_final - A4 @ state_initial)

	state0 = np.squeeze(state_initial)

	controls = np.flip(np.squeeze(u_d))

	controls = np.append(controls,[0, 0])

	nr_steps = controls.shape[0]
	nr_states = 4

	# We now compute finer dynamics and control vectors for smoother visualization
	Afine = fractional_matrix_power(A_d,(1/FPS))
	Asum = np.eye(nr_states)
	for i in range(1, FPS):
	    Asum = Asum + np.linalg.matrix_power(Afine,i)
	    
	bfine = np.linalg.inv(Asum).dot(np.squeeze(b_d))

	# We also expand the controls in the "intermediate steps" (only for visualization)
	controls_final = np.outer(controls, np.ones(FPS)).flatten()
	controls_final = np.append(controls_final, [0])

	# We compute all the states starting from x0 and using the controls
	states = np.empty([FPS*(nr_steps)+1, nr_states])
	states[0,:] = state0;
	for stepId in range(1,FPS*(nr_steps)+1):
	    states[stepId, :] = np.dot(Afine,states[stepId-1, :]) + controls_final[stepId-1] * bfine
	    
	# Now create the time vector for simulation
	t = np.linspace(1/FPS,nr_steps,FPS*(nr_steps),endpoint=True)
	t = np.append([0], t)

	return states

"""
	Calculates and returns the next state based on the current state
	and applied input. Has option for linear and non-linear state 
	variable calculation.
	INPUTS:
		cur_state - (4, ) np.array representing our current state
		u - float that represents the force applied on the system 
		linearized - boolean for choosing linear or non-linear system
	OUTPUTS:
		next_state - (4, ) np.array representing our next state
"""
def next_state(cur_state, u, linearized=False):
	# Unpack state variables
	x = cur_state[0]
	x_dot = cur_state[1]
	theta = cur_state[2]
	theta_dot = cur_state[3]

	if linearized:
		# If we are linearized, then just use Ax + bu for next state
		_, acceleration, _, ang_acceleration = A @ cur_state + b.flatten() * u
	else:
		# Calculate angular and linear acceleration
		acceleration = ((u/m) + (np.sin(theta) * l * theta_dot ** 2) - (g * np.sin(theta) * np.cos(theta)) -  \
			(x_dot * k / m)) / ((M/m) + (np.sin(theta)) ** 2)
		ang_acceleration = ((-np.cos(theta) * u / m) - (np.sin(theta) * np.cos(theta) * l * theta_dot ** 2) +  \
			((1 + M/m) * g * np.sin(theta)) + (k * x_dot * np.cos(theta)/m)) / ((M/m) + (np.sin(theta)) ** 2) / l
		
	# Update angular and linear velocities 
	vel_new = x_dot + acceleration
	ang_vel_new = theta_dot + ang_acceleration

	# Update angular and linear positions
	pos_new = x + vel_new
	ang_new = theta + ang_vel_new

	# Make sure angle is alwaus between -pi and pi
	while ang_new < -np.pi:
		ang_new += 2 * np.pi
	while (ang_new > np.pi):
		ang_new -= 2 * np.pi

	# Return next state
	next_state = np.array([pos_new, vel_new, ang_new, ang_vel_new]) 
	return next_state

"""
	Uses the arcade graphic functions to draw a cart pole 
	system given the current state variables.
	INPUTS:
		cur_state - (4, ) np.array representing our current state
		u - float that represents the force applied on the system 
		vis_inp - boolean for visualizing the input force
"""
def draw_current_state(cur_state, u, vis_inp=False):
	# Unpack state variables
	x = cur_state[0] + SCREEN_WIDTH//2
	theta = cur_state[2]

	# Check if off screen:
	if x < 0:
		x = SCREEN_WIDTH
	elif x > SCREEN_WIDTH:
		x = 0

	# Draw Text/Title
	arcade.draw_text("Linearized Continuous Time Simulation", SCREEN_WIDTH//2, 3*SCREEN_HEIGHT//4,
                         arcade.color.BLACK, 24, width=SCREEN_WIDTH, align="center",
                         anchor_x="center", anchor_y="center")
	# Draw cart
	arcade.draw_rectangle_filled(x, GROUND_HEIGHT + 2*WHEEL_RAD + CART_HEIGHT//2, CART_WIDTH, CART_HEIGHT, arcade.color.BLUSH)
	# Draw ground
	arcade.draw_rectangle_filled(SCREEN_WIDTH//2, GROUND_HEIGHT//2, SCREEN_WIDTH, GROUND_HEIGHT, arcade.color.CYBER_GRAPE)
	# Draw wheels
	arcade.draw_circle_filled(x - CART_WIDTH // 3, GROUND_HEIGHT + WHEEL_RAD, WHEEL_RAD, arcade.color.BLACK)
	arcade.draw_circle_filled(x + CART_WIDTH // 3, GROUND_HEIGHT + WHEEL_RAD, WHEEL_RAD, arcade.color.BLACK)
	# Compute pole locations
	pole_pivot_y = GROUND_HEIGHT + CART_HEIGHT + 2*WHEEL_RAD
	pole_tip_y = pole_pivot_y + l * np.cos(theta)
	pole_tip_x = x + l * np.sin(theta)
	# Draw pole
	arcade.draw_line(x, pole_pivot_y, pole_tip_x, pole_tip_y, arcade.color.BLACK, 2)
	# Draw small mass
	arcade.draw_circle_filled(pole_tip_x, pole_tip_y, BALL_RAD, arcade.color.BLACK)
	# Visualize input
	if vis_inp:
		arcade.draw_line(x, GROUND_HEIGHT + CART_HEIGHT//2 + 2*WHEEL_RAD, x + 100 * u, GROUND_HEIGHT + CART_HEIGHT//2 + 2*WHEEL_RAD, arcade.color.BLACK, 2)

"""
	Computes an input based on the current state.
	INPUTS:
		cur_state - (4, ) np.array representing our current state
	RETURNS:
		u - float that represents the force applied on the system 
"""
def update_input(cur_state):
	# Unpack state variables
	theta = cur_state[2]
	theta_dot = cur_state[3]

	# Return control stategy
	return 6 * theta - 7 * theta_dot

"""
	Built in arcade function that is called every _delta_time seconds.
"""
def draw(_delta_time):
	global states, u, count
	arcade.start_render()

	# Draws the current state
	draw_current_state(states[-1], u, True)

	# Obtain next state
	if count == 0:
		states = np.vstack((states, next_state(states[-1], u, False)))
	else:
		count -= 1


def main():
	global states
	arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
	arcade.set_background_color(arcade.color.LAVENDER)
	arcade.schedule(draw, 1 / FPS)
	arcade.run()
	arcade.close_window()
	plt.plot(energy)
	plt.show()

if __name__ == "__main__":
    main()
