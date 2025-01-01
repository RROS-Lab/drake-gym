import numpy as np
import os
import webbrowser

from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    RollPitchYaw,
    StartMeshcat,
    Integrator,
    InverseKinematics,
    LogVectorOutput,
    TrajectorySource,
    Solve,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario, MakeHardwareStation

from control.cartesian_impedance import CartesianImpedanceController
from control.psuedo_inverse import PseudoInverseController
from control.pose_traj_source import PoseTrajectorySource
from reporters.contact_state import ContactReporter
from reporters.eef_state import EEStateMonitor

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class IIWALeaf:
    def __init__(
        self,
        scenario_filepath,
        controller_type,
        peg_type,
        sim_timestep,
        traj: PiecewisePose = None,
    ):
        self.builder = DiagramBuilder()
        self.meshcat = StartMeshcat()
        web_url = self.meshcat.web_url()
        print("Meshcat available at %s" % web_url)
        scenario = LoadScenario(filename=scenario_filepath)
        self.station = self.builder.AddSystem(
            MakeHardwareStation(scenario, meshcat=self.meshcat)
        )
        self.plant = self.station.GetSubsystemByName("plant")
        self.peg_name = peg_type
        self.controller_type = controller_type
        self.sim_timestep = sim_timestep
        webbrowser.open_new(web_url)

        ##Additing IK System
        self.ik = InverseKinematics(self.plant)

        # self.differential_ik = self.builder.AddSystem(DifferentialIK(self.plant,
        #             self.plant.GetFrameByName(self.peg_name), self.diff_if_params, self.ik_time_step))
        # self.differential_ik.set_name('DifferentialIK')

        if controller_type == "position":
            ##NOTE: This Code works fine as long as the scenario.yaml has the position_only flag set for model_drivers
            if traj:
                controller_plant = self.station.GetSubsystemByName(
                    "iiwa_controller_plant_pointer_system",
                ).get()
                self.initial_pose = traj.GetPose(traj.start_time())
                traj_source = self.builder.AddSystem(PoseTrajectorySource(traj))
                self.controller = AddIiwaDifferentialIK(
                    self.builder,
                    controller_plant,
                    frame=controller_plant.GetFrameByName(peg_type),
                )

                self.builder.Connect(
                    traj_source.get_output_port(),
                    self.controller.get_input_port(0),
                )

                self.builder.Connect(
                    self.station.GetOutputPort("iiwa.state_estimated"),
                    self.controller.GetInputPort("robot_state"),
                )

                self.builder.Connect(
                    self.controller.get_output_port(),
                    self.station.GetInputPort("iiwa.position"),
                )

            else:
                iiwa_position = self.builder.AddSystem(
                    ConstantVectorSource(np.zeros(7))
                )
                self.builder.Connect(
                    iiwa_position.get_output_port(),
                    self.station.GetInputPort("iiwa.position"),
                )

        elif controller_type == "pseudo_inverse":
            self.initial_pose = traj.GetPose(traj.start_time())
            V_G_source = self.builder.AddSystem(TrajectorySource(traj.MakeDerivative()))
            V_G_source.set_name("v_WG")
            self.controller = self.builder.AddSystem(
                PseudoInverseController(self.plant, traj)
            )
            self.controller.set_name("PseudoInverseController")
            self.builder.Connect(
                V_G_source.get_output_port(), self.controller.GetInputPort("V_WG")
            )

            self.integrator = self.builder.AddSystem(Integrator(7))
            self.integrator.set_name("integrator")
            self.builder.Connect(
                self.controller.get_output_port(), self.integrator.get_input_port()
            )
            self.builder.Connect(
                self.integrator.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )
            self.builder.Connect(
                self.station.GetOutputPort("iiwa.position_measured"),
                self.controller.GetInputPort("iiwa.position"),
            )

        elif controller_type == "cart_imp":
            self.initial_pose = traj.GetPose(traj.start_time())

            self.controller = self.builder.AddSystem(
                CartesianImpedanceController(self.plant, self.peg_name, traj)
            )
            V_G_source = self.builder.AddSystem(TrajectorySource(traj.MakeDerivative()))
            V_G_source.set_name("V_WG")
            self.builder.Connect(
                V_G_source.get_output_port(), self.controller.GetInputPort("V_WG")
            )

            self.builder.Connect(
                self.controller.get_output_port(0),
                self.station.GetInputPort("iiwa.torque"),
            )

            self.builder.Connect(
                self.station.GetOutputPort("iiwa.position_measured"),
                self.controller.get_input_port(0),
            )
            self.builder.Connect(
                self.station.GetOutputPort("iiwa.velocity_estimated"),
                self.controller.get_input_port(1),
            )

            self.builder.Connect(
                self.station.GetOutputPort("iiwa.torque_external"),
                self.controller.get_input_port(2),
            )

        self.add_contact_report()
        self.setup_loggers()

        self.diagram = self.builder.Build()
        self.gripper_frame = self.plant.GetFrameByName(self.peg_name)
        self.world_frame = self.plant.world_frame()
        self.meshcat.ResetRenderMode()

    def setup_loggers(self):
        self.wrench_logger = LogVectorOutput(
            self.station.GetOutputPort("iiwa.torque_external"), self.builder
        )
        self.joint_state_logger = LogVectorOutput(
            self.station.GetOutputPort("iiwa.position_measured"), self.builder
        )
        self.joint_vel_logger = LogVectorOutput(
            self.station.GetOutputPort("iiwa.velocity_estimated"), self.builder
        )

        self.end_effector_state_monitor = self.builder.AddSystem(
            EEStateMonitor(self.plant, self.peg_name)
        )

        self.builder.Connect(
            self.station.GetOutputPort("iiwa.position_measured"),
            self.end_effector_state_monitor.get_input_port(0),
        )
        self.builder.Connect(
            self.station.GetOutputPort("iiwa.velocity_estimated"),
            self.end_effector_state_monitor.get_input_port(1),
        )

        self.builder.Connect(
            self.station.GetOutputPort("iiwa.torque_external"),
            self.end_effector_state_monitor.get_input_port(2),
        )

        return

    def add_contact_report(self):
        self.contact_reporter = self.builder.AddSystem(ContactReporter())
        self.builder.Connect(
            self.station.GetOutputPort("contact_states"),
            self.contact_reporter.get_input_port(0),
        )

        return

    def visualize_frame(self, name, X_WF, length=0.15, radius=0.006):
        """
        visualize imaginary frame that are not attached to existing bodies

        Input:
            name: the name of the frame (str)
            X_WF: a RigidTransform to from frame F to world.

        Frames whose names already exist will be overwritten by the new frame
        """
        AddMeshcatTriad(
            self.meshcat, "painter/" + name, length=length, radius=radius, X_PT=X_WF
        )

    def solveIK(self, pose):

        self.ik.AddPositionConstraint(
            frameB=self.gripper_frame,
            p_BQ=np.zeros(3),
            frameA=self.world_frame,
            p_AQ_lower=pose.translation() - 2e-4,
            p_AQ_upper=pose.translation() + 2e-4,
        )

        self.ik.AddOrientationConstraint(
            frameAbar=self.world_frame,
            R_AbarA=pose.rotation(),
            frameBbar=self.gripper_frame,
            R_BbarB=RotationMatrix(),  # Desired orientation
            theta_bound=0.01,  # Tolerance in radians
        )

        result = Solve(self.ik.prog())
        joint_angles = None
        if result.is_success():
            joint_angles = result.GetSolution(self.ik.q())
            print("Joint angles:", joint_angles)
        else:
            print("Inverse kinematics problem could not be solved.")

        return joint_angles

    def performInsertion(self):
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)
        self.hole_pose: RigidTransform = self.plant.CalcRelativeTransform(
            plant_context,
            self.world_frame,
            self.plant.GetBodyByName(
                self.peg_name.split("_")[0] + "_hole"
            ).body_frame(),
        )

        if self.controller_type == "position":
            # provide initial states
            # set the joint positions of the kuka arm
            q_hole = self.solveIK(self.initial_pose)
            station_context = self.station.GetMyContextFromRoot(context)
            iiwa = self.plant.GetModelInstanceByName("iiwa")
            self.plant.SetPositions(plant_context, iiwa, q_hole)
            self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
            # peg = self.plant.GetModelInstanceByName("peg")

        elif self.controller_type == "joint_stiffness":
            q_hole = self.solveIK(self.initial_pose)
            station_context = self.station.GetMyContextFromRoot(context)
            iiwa = self.plant.GetModelInstanceByName("iiwa")
            self.station.GetInputPort("iiwa.torque").FixValue(
                station_context, np.zeros((7, 1))
            )
            self.plant.SetPositions(plant_context, iiwa, q_hole)
            # self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))

        elif self.controller_type == "pseudo_inverse":
            q_hole = self.solveIK(self.initial_pose)
            station_context = self.station.GetMyContextFromRoot(context)
            iiwa = self.plant.GetModelInstanceByName("iiwa")
            # self.station.GetInputPort("iiwa.torque").FixValue(station_context, np.zeros((7, 1)))
            self.plant.SetPositions(plant_context, iiwa, q_hole)
            # # provide initial states
            # frame_id = self.plant.GetBodyFrameIdOrThrow(self.gripper_frame.body().index())

            # # set the joint positions of the kuka arm
            # iiwa = self.plant.GetModelInstanceByName("iiwa")
            # # self.plant.SetPositions(plant_context, iiwa, q_hole)
            # # self.plant.SetPositions(plant_context, iiwa, np.zeros(7))
            # # self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
            # desired_pose = self.target_pose  # Target pose
            # desired_pose_vector = FramePoseVector()
            # desired_pose_vector.set_value(frame_id,desired_pose)
            # self.diagram.get_input_port(0).FixValue(context,desired_pose_vector)

            self.integrator.set_integral_value(
                self.integrator.GetMyContextFromRoot(context),
                self.plant.GetPositions(
                    self.plant.GetMyContextFromRoot(context),
                    self.plant.GetModelInstanceByName("iiwa"),
                ),
            )

        elif self.controller_type == "cart_imp":
            q_hole = self.solveIK(self.initial_pose)
            station_context = self.station.GetMyContextFromRoot(context)
            iiwa = self.plant.GetModelInstanceByName("iiwa")
            # self.station.GetInputPort("iiwa.torque").FixValue(station_context, np.zeros((7, 1)))
            self.plant.SetPositions(plant_context, iiwa, q_hole)
            self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))

        # self.ee_wrench = self.station.GetOutputPort("iiwa.torque_external").Eval(plant_context)
        # print(self.ee_wrench)b

        return context

    def get_X_WG(self, context=None):
        if not context:
            context = self.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        X_WG = self.plant.CalcRelativeTransform(
            plant_context, frame_A=self.world_frame, frame_B=self.gripper_frame
        )
        return X_WG

    def simulate(self, sim_duration=7.0):
        self.context = self.performInsertion()
        simulator = Simulator(self.diagram, self.context)
        simulator.Initialize()
        simulator.set_target_realtime_rate(1.0)

        # simulator.set_monitor(self.)
        self.meshcat.StartRecording(set_visualizations_while_recording=False)
        # self.meshcat.StartRecording(frames_per_second=256.0)
        simulator.AdvanceTo(sim_duration)
        self.meshcat.StopRecording()
        self.diagram.ForcedPublish(self.context)
        self.meshcat.PublishRecording()
        print(self.end_effector_state_monitor.logged_data[-2])
        print(self.contact_reporter.logged_values[-2])
        self.meshcat.AddButton("Stop Simulation")
        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
            simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        self.meshcat.DeleteButton("Stop Simulation")
        import time

        time.sleep(5)


def compose_key_frames(num_points, initial_pose, final_pose, time_start, time_end):
    """
    returns: a list of RigidTransforms
    """
    # this is an template, replace your code below
    key_frame_poses_in_world = []
    translation_trajectory = np.array(
        [
            np.linspace(
                initial_pose.translation()[i], final_pose.translation()[i], num_points
            )
            for i in range(3)
        ]
    ).T
    time_vector = np.linspace(time_start, time_end, num_points)

    # Interpolate rotation (using slerp between initial and final rotations)
    initial_rotation = initial_pose.rotation()
    final_rotation = final_pose.rotation()
    initial_quaternion = initial_rotation.ToQuaternion()
    final_quaternion = final_rotation.ToQuaternion()

    quaternions = np.array(
        [
            initial_quaternion.slerp(t / (time_end - time_start), final_quaternion)
            for t in time_vector
        ]
    )
    for i in range(num_points):
        x = translation_trajectory[i][0]
        y = translation_trajectory[i][1]
        z = translation_trajectory[i][2]

        # Here we assume that the robot always faces along the direction of travel.
        this_pose = RigidTransform(RotationMatrix(quaternions[i]), np.array([x, y, z]))

        key_frame_poses_in_world.append(this_pose)

    return key_frame_poses_in_world, time_vector


def main():

    num_key_frames = 100
    scenario_data = os.path.join(
        MAIN_DIR, "sim_map", "assets", "environment.scenario.yaml"
    )

    Initial_Pose = RigidTransform(
        R=RotationMatrix(np.eye(3)), p=np.array([0.594, 0.0, 0.1])
    )
    Final_pose = RigidTransform(
        RollPitchYaw(np.array([0.0, np.deg2rad(3), 0.0])), p=np.array([0.594, 0.0, 0.0])
    )

    total_time = 20

    key_frame_poses_in_world, time_vector = compose_key_frames(
        num_key_frames, Initial_Pose, Final_pose, 0, total_time
    )
    traj = PiecewisePose.MakeLinear(time_vector, key_frame_poses_in_world)
    timestep = total_time / (num_key_frames + 1)

    iiwaLeaf = IIWALeaf(
        scenario_filepath=scenario_data,
        controller_type="cart_imp",
        peg_type="cross_peg",
        traj=traj,
        sim_timestep=timestep,
    )

    iiwaLeaf.simulate(sim_duration=total_time)
    # Plot the results.

    return


if __name__ == "__main__":
    main()
