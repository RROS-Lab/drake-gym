from pydrake.geometry import StartMeshcat
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.analysis import Simulator
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig
from pydrake.common.value import (
    Value,
)  # FIXME(dhanush): Is this replaced by AbstractValue?
from pydrake.multibody.plant import ContactResults
from pydrake.systems.framework import LeafSystem

##Controller Params
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.all import ConstantVectorSource

import os
from pydrake.systems.framework import DiagramBuilder, PortDataType, BasicVector
import numpy as np


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
meshcat = StartMeshcat()


def clear_meshcat():
    # Clear MeshCat window from the previous blocks.
    meshcat.Delete()
    meshcat.DeleteAddedControls()


def add_scene(compliant_peg, hole, peg_frame, hole_frame, table_top, time_step=1e-3):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlant(
        MultibodyPlantConfig(
            time_step=time_step, discrete_contact_approximation="similar"
        ),
        builder,
    )
    parser = Parser(plant)

    # Load the table top and the box we created.
    # parser.AddModels(table_top)

    ##Load the pegs

    # parser.AddModelsFromString(compliant_peg, "sdf")
    # parser.AddModelsFromString(hole, "sdf")
    # parser.AddModels(compliant_peg)
    # parser.AddModels(hole)

    iiwa_url = os.path.join(
        MAIN_DIR, "models-master/iiwa_description/sdf/iiwa14_polytope_collision.sdf"
    )
    parser.AddModels(iiwa_url)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Weld the rigid box to the world so that it's fixed during simulation.
    # The top surface passes the world's origin.
    # plant.WeldFrames(plant.world_frame(),
    #                  plant.GetFrameByName(hole_frame))

    ##Adding Gravity
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.8])

    # Finalize the plant after loading the scene.
    plant.Finalize()

    # Set how high the center of the compliant box is from the world's origin.
    # W = the world's frame
    # C = frame at the center of the compliant box
    # Hole_T_Peg = RigidTransform(p=[0, 0, 0.02], R=RotationMatrix(np.eye(3)))
    # plant.SetDefaultFreeBodyPose(plant.GetBodyByName(peg_frame), Hole_T_Peg)

    return builder, plant, scene_graph


def add_viz(builder, plant):
    ApplyVisualizationConfig(
        config=VisualizationConfig(publish_period=1 / 256.0, publish_contacts=False),
        builder=builder,
        meshcat=meshcat,
    )

    return builder, plant


class custom_leaf_system(LeafSystem):
    def __init__(self, num_in, num_out):
        LeafSystem.__init__(self)
        input_port = self.DeclareInputPort(
            "input",
            PortDataType.kVectorValued,
            num_in,
        )

        # The output is re-arranging the input:
        def output(context, output):
            x = input_port.Eval(context)
            y = np.array([x[1], x[0]], dtype=float)
            output.set_value(y)

        output_port = self.DeclareVectorOutputPort(
            "output",
            BasicVector(num_out),
            output,
        )


class ContactReporter(LeafSystem):
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.
        self.DeclareAbstractInputPort(
            name="contact_results",
            model_value=Value(
                # Input port will take ContactResults from MultibodyPlant
                ContactResults()
            ),
        )
        # Calling `ForcedPublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        print()
        print(f"ContactReporter::Publish() called at time={context.get_time()}")
        contact_results = self.get_input_port().Eval(context)

        num_hydroelastic_contacts = contact_results.num_hydroelastic_contacts()
        print(f"num_hydroelastic_contacts() = {num_hydroelastic_contacts}")

        for c in range(num_hydroelastic_contacts):
            print()
            print(f"hydroelastic_contact_info({c}): {c}-th hydroelastic contact patch")
            hydroelastic_contact_info = contact_results.hydroelastic_contact_info(c)

            spatial_force = hydroelastic_contact_info.F_Ac_W()
            print(
                "F_Ac_W(): spatial force (on body A, at centroid of contact surface, in World frame) = "
            )
            print(f"{spatial_force}")

            print("contact_surface()")
            contact_surface = hydroelastic_contact_info.contact_surface()
            num_faces = contact_surface.num_faces()
            total_area = contact_surface.total_area()
            centroid = contact_surface.centroid()
            print(f"total_area(): area of contact surface in m^2 = {total_area}")
            print(f"num_faces(): number of polygons or triangles = {num_faces}")
            print(f"centroid(): centroid (in World frame) = {centroid}")


def add_contact_report(builder, plant):
    contact_reporter = builder.AddSystem(ContactReporter())
    builder.Connect(
        plant.get_contact_results_output_port(), contact_reporter.get_input_port(0)
    )

    return builder, plant


def add_controller(builder, plant, scene_graph, target_axis_index, K_p, K_d, K_i):

    desired_state = ConstantVectorSource(np.zeros(plant.num_multibody_states()))

    desired_state_source = builder.AddSystem(desired_state)
    desired_state_source.set_name("constant_source")

    num_actuators = plant.num_actuators()
    kp = np.ones(num_actuators) * K_p
    kd = np.ones(num_actuators) * K_d
    ki = np.ones(num_actuators) * K_i
    inv_dyn_controller = InverseDynamicsController(
        plant, kp=kp, kd=kd, ki=ki, has_reference_acceleration=False
    )

    builder.AddSystem(inv_dyn_controller)

    builder.Connect(
        inv_dyn_controller.get_output_port_control(),
        plant.get_applied_generalized_force_input_port(),
    )
    builder.Connect(
        plant.get_state_output_port(),
        inv_dyn_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        desired_state_source.get_output_port(),
        inv_dyn_controller.get_input_port_desired_state(),
    )

    ##Adding constant zero load
    constant_zero_torque = ConstantVectorSource(np.zeros(num_actuators))
    builder.AddSystem(constant_zero_torque)
    builder.Connect(
        constant_zero_torque.get_output_port(), plant.get_actuation_input_port()
    )
    print(plant.get_source_id())
    # builder.Connect(plant.get_geometry_pose_output_port(),
    #                 scene_graph.get_source_pose_port(plant.get_source_id()))
    # builder.Connect(scene_graph.get_query_output_port(),
    #                 plant.get_geometry_query_input_port())

    print("Done")
    return builder, plant


def add_contact_viz(builder, plant):
    contact_viz = ContactVisualizer.AddToBuilder(
        builder,
        plant,
        meshcat,
        ContactVisualizerParams(
            publish_period=1.0 / 256.0,
            newtons_per_meter=2e1,
            newton_meters_per_meter=1e-1,
        ),
    )

    return builder, plant


def add_subsystems(builder, plant, scene_graph, viz=True):
    clear_meshcat()

    if viz:
        add_viz(builder, plant)

    add_contact_report(builder, plant)

    if viz:
        add_contact_viz(builder, plant)

    ##Add Controller
    # Add impedance controller
    target_axis_index = 0  # Example: controlling along the x-axis
    K_p, K_d, K_i = 10.0, 5.0, 0.1
    builder, plant = add_controller(
        builder, plant, scene_graph, target_axis_index, K_p, K_d, K_i
    )

    diagram = builder.Build()
    print("Starting Simulator")

    return diagram


def send_to_joint_state(plant, diagram, joint_state):
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    iiwa = plant.GetModelInstanceByName("iiwa14")
    plant.SetPositions(plant_context, iiwa, joint_state)
    # plant.SetVelocities(plant_context, iiwa, np.zeros(7))
    # wsg = plant.GetModelInstanceByName("wsg")
    # plant.SetPositions(plant_context, wsg, [-0.05, 0.05])
    # plant.SetVelocities(plant_context, wsg, [0, 0])
    # plant.SetPositions(joint_state)

    return plant, diagram


def run_simulation(diagram, sim_time):

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(10.0)
    simulator.Initialize()
    meshcat.StartRecording(frames_per_second=30.0)
    simulator.AdvanceTo(sim_time)
    meshcat.StopRecording()

    # Numerically report contact results at the end of simulation.
    diagram.ForcedPublish(simulator.get_context())

    return


def run_peg_hole_sim():
    peg_url = os.path.join(MAIN_DIR, "assets", "square_peg", "square_peg.sdf")
    hole_url = os.path.join(MAIN_DIR, "assets", "square_peg", "square_hole.sdf")
    table_top = os.path.join(MAIN_DIR, "assets", "table_top.sdf")
    clear_meshcat()
    builder, plant, scene_graph = add_scene(
        peg_url, hole_url, "square_peg", "square_hole", table_top
    )
    diagram = add_subsystems(builder, plant, scene_graph)

    q0 = np.array(
        [
            0.1,
            0.5,
            0.3,
            -1.32296976e00,
            -6.29097287e-06,
            1.61181157e00,
            -2.66900985e-05,
        ]
    )

    plant, diagram = send_to_joint_state(plant, diagram, q0)

    run_simulation(diagram, sim_time=6)
    meshcat.PublishRecording()
    # add_viz(builder, plant)
    # simulator = Simulator(builder.Build())
    # simulator.AdvanceTo(0)
    # meshcat.PublishRecording()
    import time

    time.sleep(10)


if __name__ == "__main__":
    run_peg_hole_sim()
