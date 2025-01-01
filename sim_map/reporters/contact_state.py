from pydrake.multibody.plant import ContactResults
from pydrake.systems.framework import LeafSystem
from pydrake.common.value import (
    Value,
)  # FIXME(dhanush): Is this replaced by AbstractValue?


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
        self.logged_values = []
        self.DeclarePerStepPublishEvent(self.Publish)

    def Publish(self, context):
        # print(f"ContactReporter::Publish() called at time={context.get_time()}")
        contact_results = self.get_input_port().Eval(context)

        num_hydroelastic_contacts = contact_results.num_hydroelastic_contacts()
        # print(f"num_hydroelastic_contacts() = {num_hydroelastic_contacts}")

        for c in range(num_hydroelastic_contacts):
            log_dict = {}

            # print(f"hydroelastic_contact_info({c}): {c}-th hydroelastic contact patch")
            hydroelastic_contact_info = contact_results.hydroelastic_contact_info(c)

            spatial_force = hydroelastic_contact_info.F_Ac_W()
            # print("F_Ac_W(): spatial force (on body A, at centroid of contact surface, in World frame) = ")
            # print(f"{spatial_force}")

            # print("contact_surface()")
            contact_surface = hydroelastic_contact_info.contact_surface()
            num_faces = contact_surface.num_faces()
            total_area = contact_surface.total_area()
            centroid = contact_surface.centroid()
            log_dict["time"] = context.get_time()
            log_dict["num_faces"] = num_faces
            log_dict["total_area"] = total_area
            log_dict["centroid"] = centroid
            log_dict["spatial_force"] = spatial_force

            self.logged_values.append(log_dict)
            # print(f"total_area(): area of contact surface in m^2 = {total_area}")
            # print(f"num_faces(): number of polygons or triangles = {num_faces}")
            # print(f"centroid(): centroid (in World frame) = {centroid}")
