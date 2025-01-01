import numpy as np

from pydrake.all import (
    LeafSystem,
    JacobianWrtVariable,
    AngleAxis,
)


class PseudoInverseController(LeafSystem):
    def __init__(self, plant, trajectory):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("square_peg").body_frame()
        self._W = plant.world_frame()

        self.commanded_traj = trajectory

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        # J_G = J_G[:, self.iiwa_start : self.iiwa_end + 1]  # Only iiwa terms.

        # Compute pose errors
        desired_pose = self.commanded_traj.GetPose(context.get_time())
        current_pose = self._plant.CalcRelativeTransform(
            self._plant_context, self._plant.world_frame(), self._G
        )

        error_translation = desired_pose.translation() - current_pose.translation()
        R_error = desired_pose.rotation() @ current_pose.rotation().inverse()
        error_vector = (
            AngleAxis(R_error.matrix()).axis() * AngleAxis(R_error.matrix()).angle()
        )

        print(error_translation)
        error = np.concatenate(error_translation, error_vector)
        v = np.linalg.pinv(J_G).dot(V_G)

        output.SetFromVector(v)
