"""
One-dimensional grid delay task, with continuous sensor value.

Similar to grid_1D_delay with one important difference.
The sensor value is a single float, rather than an array
of binary.
In order to succesfully learn this task, becca must first
discretize the input.

To run this world from the command line

    python3 grid_1D_delay_cont

"""
import becca.brain as becca_brain
from becca_test.grid_1D_delay import World as Grid_1D_Delay_World


class World(Grid_1D_Delay_World):
    """
    One-dimentional delayed grid task with continous sensing.
    """
    def __init__(self, lifespan=None):
        Grid_1D_Delay_World.__init__(self, lifespan)
        self.name = 'grid_1D_delay_continuous'
        print('  -- continuous sensor')
        self.n_sensors = 1
        self.visualize_interval = 1e6

    def sense(self):
        return [self.world_state]


if __name__ == "__main__":
    becca_brain.run(World())
