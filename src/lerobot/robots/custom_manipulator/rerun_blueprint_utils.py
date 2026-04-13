import rerun as rr
import rerun.blueprint as rrb


def send_custom_manipulator_blueprint():
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Panda",
                    origin="/",
                    contents=["/panda/**", "/state_eef/**", "/target_eef/**"],
                    spatial_information=rrb.SpatialInformation(
                        target_frame="tf#/panda/panda_link0",
                        show_axes=True,
                        show_bounding_box=True,
                    ),
                ),
                rrb.Spatial3DView(
                    name="xHand",
                    origin="/",
                    contents=["/xhand/**", "/tips/**", "/targets/**", "/forces/**"],
                    spatial_information=rrb.SpatialInformation(
                        target_frame="tf#/xhand/right_hand_link",
                        show_axes=True,
                        show_bounding_box=True,
                    ),
                ),
            ),
        ),
    )
