import numpy as np

from pydrake.systems.framework import LeafSystem
from pydrake.all import JacobianWrtVariable


class EEStateMonitor(LeafSystem):
    def __init__(self, plant, peg_name):
        super().__init__()
        self._plant = plant
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._plant_context = plant.CreateDefaultContext()
        self._G = plant.GetBodyByName(peg_name).body_frame()
        self._W = plant.world_frame()

        self.DeclareVectorInputPort("joint_positions", plant.num_positions())
        self.DeclareVectorInputPort("joint_velocities", plant.num_velocities())
        self.DeclareVectorInputPort("joint_torques", plant.num_actuators())
        self.logged_data = []
        self.DeclarePerStepPublishEvent(self._log_state)

    def _log_state(self, context):
        # Get input data
        joint_positions = self.get_input_port(0).Eval(context)
        joint_velocities = self.get_input_port(1).Eval(context)
        joint_torques = self.get_input_port(2).Eval(context)

        # Set joint states in the plant context
        plant_context = self._plant.CreateDefaultContext()
        self._plant.SetPositions(plant_context, joint_positions)
        self._plant.SetVelocities(plant_context, joint_velocities)

        # Compute end-effector state
        end_effector_frame = self._plant.GetBodyByName("square_peg").body_frame()

        # Position
        position = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), end_effector_frame
        ).translation()

        orientation = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), end_effector_frame
        ).rotation()
        # Velocity
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )

        # J_G = J_G[:, 0:7] ##Add this if you want flange
        spatial_velocity = J_G.dot(joint_velocities)

        # Wrench
        end_effector_wrench = np.linalg.pinv(np.transpose(J_G)).dot(joint_torques)

        # Log data
        state_data = {
            "time": context.get_time(),
            "position": position,
            "orientation": orientation,
            "velocity": spatial_velocity,
            "wrench": end_effector_wrench,
        }
        self.logged_data.append(state_data)
