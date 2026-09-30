import os

# The tests that create nodes must not talk to a simulation running on this machine:
# they would receive its /nav_status and could publish on its /cmd_vel.
os.environ['ROS_DOMAIN_ID'] = '97'
