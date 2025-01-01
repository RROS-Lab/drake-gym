import numpy as np

from pydrake.all import (
    LeafSystem,
    RigidTransform,
    JacobianWrtVariable,
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsParameters,
)


class CartesianImpedanceController(LeafSystem):

    def __init__(self, plant, peg_name, commanded_trajectory):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa_model = plant.GetModelInstanceByName("iiwa")
        self._TOOL_FRAME = plant.GetBodyByName(peg_name).body_frame()
        self._WORLD_FRAME = plant.world_frame()
        self.commanded_trajectory = commanded_trajectory
        self.it = 0
        self.define_controller_params()
        self.DeclareVectorInputPort("iiwa_position_measured", 7)
        self.DeclareVectorInputPort("iiwa_velocity_measured", 7)

        # If we want, we can add this in to do closed-loop force control on z.
        self.DeclareVectorInputPort("iiwa_torque_external", 7)
        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)

        self.DeclareVectorOutputPort("iiwa_torque_cmd", 7, self.CalcTorqueOutput)

    def define_controller_params(self):

        self.diff_ik_params = DifferentialInverseKinematicsParameters(7, 7)
        self.ik_time_step = self._plant.time_step()
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])

        factor = 1.0  # velocity limit factor
        self.diff_ik_params.set_joint_velocity_limits(
            (-factor * iiwa14_velocity_limits, factor * iiwa14_velocity_limits)
        )

        self.translation_stiffness = np.asarray([200.0, 200.0, 200.0])
        self.orientation_stiffness = np.asarray([10.0, 10.0, 10.0])

        self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
        self.damping_ratio = 1.0

        damping_pos = self.damping_ratio * 5 * np.sqrt(self.translation_stiffness)
        damping_ori = self.damping_ratio * 5 * np.sqrt(self.orientation_stiffness)

        ##Controller Gains
        self.Kp = np.concatenate(
            [self.translation_stiffness, self.orientation_stiffness], axis=0
        )
        self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)
        self.q0 = None
        return

    def CalcDifferentialIK(self, q_last, current_pose: RigidTransform):

        x = self._plant.GetPositionsAndVelocities(self._plant_context)
        x[: self._plant.num_positions()] = q_last
        result = DoDifferentialInverseKinematics(
            self._plant,
            self._plant_context,
            current_pose,
            self._TOOL_FRAME,
            self.diff_ik_params,
        )

        joint_angles = np.zeros(7)
        if result.status != result.status.kSolutionFound:
            print("Differential IK could not find a solution.")
        else:
            joint_angles = q_last + self.ik_time_step * result.joint_velocities

        return joint_angles

    def CalcTorqueOutput(self, context, output):

        # Read inputs
        q_now = self.get_input_port(0).Eval(
            context
        )  # current joint position, Shape (n_joints, )
        v_now = self.get_input_port(1).Eval(
            context
        )  # current joint velocity, Shape (n_joints, )
        tau_external = self.get_input_port(2).Eval(
            context
        )  # external torque, Shape (n_joints, )

        # manually update ?
        self._plant.SetPositions(self._plant_context, self._iiwa_model, q_now)
        self._plant.SetVelocities(self._plant_context, self._iiwa_model, v_now)

        # 1. Convert joint space quantities to Cartesian quantities.
        X_now = self._plant.CalcRelativeTransform(
            self._plant_context, self._WORLD_FRAME, self._TOOL_FRAME
        )  # type -> pydrake.math.RigidTransform

        p_WE = X_now.translation()
        R_WE = X_now.rotation()

        J_spatial = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._TOOL_FRAME,
            [0, 0, 0],
            self._WORLD_FRAME,
            self._WORLD_FRAME,
        )
        # J_spatial = J_spatial[:, :7]

        X_WE = self._plant.CalcRelativeTransform(
            self._plant_context, self._WORLD_FRAME, self._TOOL_FRAME
        )
        p_WE = X_WE.translation()
        R_WE = X_WE.rotation()

        # Calculate pose error
        desired_pose = self.commanded_trajectory.GetPose(context.get_time())
        ##Calculating Desired Pose
        if self.it == 0:
            self.q_last = q_now

        q_desired = self.CalcDifferentialIK(self.q_last, desired_pose)
        self.q_last = q_now

        p_error = desired_pose.translation() - p_WE
        self.it += 1
        quat_error = (
            desired_pose.rotation()
            .ToQuaternion()
            .multiply(R_WE.ToQuaternion().inverse())
        )
        orientation_error = quat_error.xyz() * np.sign(quat_error.w())

        pose_error = -np.concatenate(
            [orientation_error, p_error]
        )  # to make it (current-desired)

        if False:
            M = self._plant.CalcMassMatrix(self._plant_context)
            # Calculate operational space inertia matrix
            if abs(np.linalg.det(J_spatial @ np.linalg.inv(M) @ J_spatial.T)) >= 1e-2:
                Lambda = np.linalg.inv(J_spatial @ np.linalg.inv(M) @ J_spatial.T)
            else:
                Lambda = np.linalg.pinv(
                    J_spatial @ np.linalg.inv(M) @ J_spatial.T, rcond=1e-2
                )

            F_desired = Lambda @ a_desired

        # Compute generalized forces.
        cart_vel = J_spatial @ v_now

        # Calculate desired acceleration
        a_desired = -self.Kp * pose_error - self.Kd * cart_vel

        # Add joint task in nullspace.
        tau_0 = -self.Kp_null * (q_now - q_desired) - self.Kd_null * v_now
        tau_null = (np.eye(7) - J_spatial.T @ np.linalg.pinv(J_spatial.T)) @ tau_0

        # Calculate bias forces (Coriolis, Centripetal, Gyroscopic)
        tau_bias = self._plant.CalcBiasTerm(self._plant_context)

        # composition of torque
        tau = J_spatial.T @ a_desired + tau_bias + tau_external + tau_null

        # Gravity Compensation
        # if self._plant.is_gravity_enabled(self._iiwa_model):
        #     tau += self._plant.CalcGravityGeneralizedForces(self._plant_context)

        np.clip(tau, -100, 100)
        output.SetFromVector(tau)
