import numpy as np

from pydrake.all import (
    AbstractValue,
    LeafSystem,
    JacobianWrtVariable,
    FramePoseVector,
    AngleAxis,
)


class PseudoInversePositionController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("square_peg").body_frame()
        self._W = plant.world_frame()

        self.K_p = 0.5
        self.K_o = 0.2

        self.DeclareVectorInputPort("iiwa.position", 7)
        # self.DeclareVectorInputPort("commanded_pose", 7)
        self.DeclareAbstractInputPort(
            "desired_pose", AbstractValue.Make(FramePoseVector())
        )
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.get_input_port(0).Eval(context)
        # q_commanded = self.get_input_port(1).Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )

        # J_G = J_G[:, 0:7]  # Ignore gripper terms

        # Compute pose errors
        frame_id = self._plant.GetBodyFrameIdOrThrow(self._G.body().index())
        desired_pose = self.get_input_port(1).Eval(context).value(frame_id)
        current_pose = self._plant.CalcRelativeTransform(
            self._plant_context, self._plant.world_frame(), self._G
        )

        if False:
            J_G = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context,
                with_respect_to=JacobianWrtVariable.kQDot,
                frame_B=self._G,
                p_BoBp_B=current_pose.translation(),
                frame_A=self._plant.world_frame(),
                frame_E=self._plant.world_frame(),
            )

        error_translation = desired_pose.translation() - current_pose.translation()
        R_error = desired_pose.rotation() @ current_pose.rotation().inverse()
        print(desired_pose, current_pose)
        error_vector = (
            AngleAxis(R_error.matrix()).axis() * AngleAxis(R_error.matrix()).angle()
        )
        # Cartesian velocity command
        cartesian_velocity = np.hstack(
            (
                self.K_p * error_translation,
                self.K_o * error_vector,
            )
        )

        v = np.linalg.pinv(J_G).dot(cartesian_velocity)
        output.SetFromVector(v)
