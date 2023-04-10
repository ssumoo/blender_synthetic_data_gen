from pathlib import Path
import json
import numpy as np
import mathutils
import bpy


REPLACE_TOKEN = "{stem}"


def copy_objects(from_col, to_col, linked, dupe_lut):
    for o in from_col.objects:
        dupe = o.copy()
        if not linked and o.data:
            dupe.data = dupe.data.copy()
        to_col.objects.link(dupe)
        dupe_lut[o] = dupe


def copy_collection_by_name(collection_name: str, new_collection_name: str, linked=False):
    def _copy(_parent, _collection, _linked=False):
        cc = bpy.data.collections.new(new_collection_name)
        copy_objects(_collection, cc, _linked, dupe_lut)
        for c in collection.children:
            _copy(cc, c, linked)
        parent.children.link(cc)
    parent = bpy.context.scene.collection
    collection = bpy.data.collections[collection_name]
    dupe_lut = {}
    _copy(parent, collection, linked)
    for o, dupe in tuple(dupe_lut.items()):
        parent = dupe_lut.get(o.parent)
        if parent:
            dupe.parent = parent


def copy_player_and_freeze_at_frame(frame, collection_name: str, new_collection_name: str):
    current_frame_idx = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(frame)
    copy_collection_by_name(collection_name, new_collection_name)
    for obj in bpy.data.collections[new_collection_name].all_objects:
        if "Armature" in obj.modifiers:
            with bpy.context.temp_override(object=obj):
                bpy.ops.object.modifier_apply(modifier="Armature")
    bpy.context.scene.frame_set(current_frame_idx)


def camera_view_bounds_2d(scene, cam_ob, me_obs):
    mat = cam_ob.matrix_world.normalized().inverted()
    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != "ORTHO"
    lx = []
    ly = []
    for me_ob in me_obs:
        me = me_ob.to_mesh()
        me.transform(me_ob.matrix_world)
        me.transform(mat)
        for v in me.vertices:
            co_local = v.co
            z = -co_local.z
            if camera_persp:
                if z <= 0.0:
                    continue
                else:
                    frame = [(v / (v.z / z)) for v in frame]
            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y
            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)
            lx.append(x)
            ly.append(y)
    if len(lx) == 0 and len(ly) == 0:
        return False, (None, None, None, None)
    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)
    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    found = ((max_y-min_y) > 1e-5) and ((max_x-min_x) > 1e-5)
    return found, (min_x*dim_x, r.resolution_y-max_y*dim_y, max_x*dim_x, r.resolution_y-min_y*dim_y)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def select_all_meshes_in_collection(collection_name: str):
    bpy.ops.object.select_all(action="DESELECT")
    collection = bpy.data.collections[collection_name]
    for obj in collection.all_objects:
        bpy.data.objects[obj.name].select_set(True)


def delete_collection_by_name(collection_name: str):
    collection = bpy.data.collections[collection_name]
    for obj in collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


class GenerationSettings(bpy.types.PropertyGroup):
    min_players: bpy.props.IntProperty(name="min_players", default=11, min=0, max=11)
    max_players: bpy.props.IntProperty(name="max_players", default=11, min=0, max=11)
    min_player_scale: bpy.props.FloatProperty(name="min_player_scale", default=0.9, min=0.0, max=3.0)
    max_player_scale: bpy.props.FloatProperty(name="max_player_scale", default=1.2, min=0.0, max=3.0)
    min_player_hue: bpy.props.FloatProperty(name="min_player_hue", default=0, min=0, max=1.0)
    max_player_hue: bpy.props.FloatProperty(name="max_player_hue", default=0, min=0, max=1.0)
    min_player_sat: bpy.props.FloatProperty(name="min_player_sat", default=0, min=0, max=1.0)
    max_player_sat: bpy.props.FloatProperty(name="max_player_sat", default=0, min=0, max=1.0)
    min_player_val: bpy.props.FloatProperty(name="min_player_val", default=0, min=0, max=1.0)
    max_player_val: bpy.props.FloatProperty(name="max_player_val", default=0, min=0, max=1.0)
    min_player_skin_hue: bpy.props.FloatProperty(name="min_player_skin_hue", default=0.45, min=0, max=1.0)
    max_player_skin_hue: bpy.props.FloatProperty(name="max_player_skin_hue", default=0.55, min=0, max=1.0)
    min_player_skin_sat: bpy.props.FloatProperty(name="min_player_skin_sat", default=0.8, min=0, max=1.0)
    max_player_skin_sat: bpy.props.FloatProperty(name="max_player_skin_sat", default=1.2, min=0, max=1.0)
    min_player_skin_val: bpy.props.FloatProperty(name="min_player_skin_val", default=0.1, min=0, max=1.0)
    max_player_skin_val: bpy.props.FloatProperty(name="max_player_skin_val", default=2.0, min=0, max=1.0)

    min_light_intensity: bpy.props.IntProperty(name="min_light_intensity", default=350000, min=0, max=2000000)
    max_light_intensity: bpy.props.IntProperty(name="max_light_intensity", default=500000, min=0, max=2000000)
    min_light_cone_angle_deg: bpy.props.IntProperty(name="min_light_cone_angle_deg", default=0, min=0, max=90)
    max_light_cone_angle_deg: bpy.props.IntProperty(name="max_light_cone_angle_deg", default=45, min=45, max=90)

    min_field_hue: bpy.props.FloatProperty(name="min_field_hue", default=0.4, min=0, max=2.0)
    max_field_hue: bpy.props.FloatProperty(name="max_field_hue", default=0.6, min=0, max=2.0)
    min_field_sat: bpy.props.FloatProperty(name="min_field_sat", default=0.8, min=0, max=2.0)
    max_field_sat: bpy.props.FloatProperty(name="max_field_sat", default=1.2, min=0, max=2.0)
    min_field_val: bpy.props.FloatProperty(name="min_field_val", default=0.4, min=0, max=2.0)
    max_field_val: bpy.props.FloatProperty(name="max_field_val", default=0.8, min=0, max=2.0)

    shade_chance: bpy.props.FloatProperty(name="shade_chance", default=0.25, min=0, max=1.0)
    shade_min: bpy.props.FloatProperty(name="shade_min", default=0.0, min=0, max=1.0)
    shade_max: bpy.props.FloatProperty(name="shade_max", default=0.5, min=0, max=1.0)

    camera_min_z: bpy.props.FloatProperty(name="camera_min_z", default=2.0, min=0)
    camera_max_z: bpy.props.FloatProperty(name="camera_max_z", default=50.0, min=0)

    save_dir: bpy.props.StringProperty(name="save_dir", subtype="FILE_PATH")
    save_format: bpy.props.StringProperty(name="save_format", default=f"img_{REPLACE_TOKEN}.png")
    images_per_config: bpy.props.IntProperty(name="images_per_config", default=10, min=0)
    n_images: bpy.props.IntProperty(name="n_images", default=10, min=0)


# {category}_{type}_{name}
class TakeRenderedImages(bpy.types.Operator):
    bl_idname = "render.take_rendered_image"
    bl_label = "Take Images"
    bl_options = {"REGISTER", "UNDO"}

    KEYPOINT_PREFIX = "kp_"

    # TODO: take this from blender
    FIELD_X_BOUNDS = 110.0
    FIELD_Y_BOUNDS = 60.0
    SHADE_ACTIVE_Z = 70.0
    MAX_PLAYER_ANIMATION_FRAMES = 100

    @classmethod
    def poll(cls, context):
        return context.active_object is not None
        
    def execute(self, context):
        print("starting data generation")
        args: GenerationSettings = context.scene.data_gen_settings
        num_configs = int(np.ceil(args.n_images / args.images_per_config))
        bpy.context.space_data.shading.type = "RENDERED"
        for i_config in range(num_configs):
            created_collection_names = []
            created_collection_names += self._place_players_on_field(args, "team_1")
            created_collection_names += self._place_players_on_field(args, "team_2")
            for i_image in range(args.images_per_config):
                ball_loc = self._randomise_ball_location(args)
                file_stem = f"cfg{str(i_config).zfill(4)}_img{str(i_image).zfill(4)}"
                self._randomise_field(args)
                self._randomise_lights(args)
                self._randomise_shades(args)
                self._sample_camera_looking_at_ball(args, ball_loc)
                self._save_img_and_ann(context, args, file_stem)
            for c in created_collection_names:
                delete_collection_by_name(c)
        return {"FINISHED"}

    def _randomise_field(self, args: GenerationSettings):
        hsv = np.random.uniform(
            [args.min_field_hue, args.min_field_sat, args.min_field_val],
            [args.max_field_hue, args.max_field_sat, args.max_field_val],
        )
        stripe_base_intensity = np.random.uniform(0.8, 1.0)
        stripe_x_intensity = np.random.uniform(0.2, 0.4)
        stripe_y_intensity = np.random.uniform(0.2, 0.4)
        if np.random.binomial(n=1, p=0.5):
            x_wave_scale = np.random.uniform(1.5, 5.0)
        else:
            x_wave_scale = 0
        if np.random.binomial(n=1, p=0.5):
            y_wave_scale = np.random.uniform(1.0, 4.0)
        else:
            y_wave_scale = 0
        bpy.data.materials["ground"].node_tree.nodes["stripe_base_intensity"].outputs[0].default_value = stripe_base_intensity
        bpy.data.materials["ground"].node_tree.nodes["stripe_x_intensity"].outputs[0].default_value = stripe_x_intensity
        bpy.data.materials["ground"].node_tree.nodes["stripe_y_intensity"].outputs[0].default_value = stripe_y_intensity
        bpy.data.materials["ground"].node_tree.nodes["x_wave_texture"].inputs[1].default_value = x_wave_scale
        bpy.data.materials["ground"].node_tree.nodes["y_wave_texture"].inputs[1].default_value = y_wave_scale
        bpy.data.materials["ground"].node_tree.nodes["grass_hsv"].inputs[0].default_value = hsv[0]
        bpy.data.materials["ground"].node_tree.nodes["grass_hsv"].inputs[1].default_value = hsv[1]
        bpy.data.materials["ground"].node_tree.nodes["grass_hsv"].inputs[2].default_value = hsv[2]

    def _randomise_ball_location(self, args: GenerationSettings):
        ball_loc = np.random.uniform(
            [-self.FIELD_X_BOUNDS*0.5, -self.FIELD_Y_BOUNDS*0.5, 0.3],
            [self.FIELD_X_BOUNDS*0.5, self.FIELD_Y_BOUNDS*0.5, 3],
        )
        bpy.data.objects["Soccer Ball"].location = ball_loc
        return ball_loc

    def _randomise_lights(self, args: GenerationSettings):
        intensity = np.random.randint(args.min_light_intensity, args.max_light_intensity)
        ang_x = np.random.uniform(args.min_light_cone_angle_deg, args.max_light_cone_angle_deg) / 180.0 * np.pi
        ang_y = np.random.uniform(args.min_light_cone_angle_deg, args.max_light_cone_angle_deg) / 180.0 * np.pi
        if np.random.binomial(n=1, p=0.5):
            ang_x *= -1
        if np.random.binomial(n=1, p=0.5):
            ang_y *= -1
        bpy.data.objects["Sun"].data.energy = intensity
        bpy.data.objects["Sun"].rotation_euler[0] = ang_x
        bpy.data.objects["Sun"].rotation_euler[1] = ang_y

    def _randomise_shades(self, args: GenerationSettings):
        shade_loc = np.array([0.0, 0.0, -10.0])
        shade_plate = bpy.data.objects["Shade Plate"]
        if not np.random.binomial(n=1, p=args.shade_chance):
            shade_plate.location = shade_loc
            return
        shade_mode = np.random.choice(["+x", "-x", "+y", "-y"])
        shade_pct = np.random.uniform(args.shade_min, args.shade_max)
        if "x" in shade_mode:
            shade_x = 0.5 * shade_plate.dimensions.x + (0.5 - shade_pct) * self.FIELD_X_BOUNDS
            shade_loc = np.array([shade_x, 0.0, self.SHADE_ACTIVE_Z])
        else:
            shade_y = 0.5 * shade_plate.dimensions.y + (0.5 - shade_pct) * self.FIELD_Y_BOUNDS
            shade_loc = np.array([0.0, shade_y, self.SHADE_ACTIVE_Z])
        if "-" in shade_mode:
            shade_loc[:2] *= -1
        shade_plate.location = shade_loc

    def _place_players_on_field(self, args: GenerationSettings, team_name: str):
        def randomise_to_black_and_white(hsv, hsv_noise=None, p_black: float = 0.1, p_white: float = 0.1):
            if np.random.binomial(1, p=p_white):
                return np.array([0.0, 1.0, 1.0, 1.0])
            elif np.random.binomial(1, p=p_black):
                return np.array([0.0, 0.0, 0.0, 1.0])
            else:
                if hsv_noise is None:
                    hsv_noise = np.zeros(4)
                return hsv + np.random.uniform(-hsv_noise, hsv_noise)

        collection_name = f"player_mesh_{team_name}"
        clothes_hvs = np.random.uniform(
            [args.min_player_hue, args.min_player_sat, args.min_player_val, 1.0],
            [args.max_player_hue, args.max_player_sat, args.max_player_val, 1.0],
        )
        hsva_noise = np.array([0.1, 0.1, 0.1, 0.0])
        shirt_hsva = randomise_to_black_and_white(clothes_hvs, hsva_noise, 0.1, 0.1)
        shorts_hsva = randomise_to_black_and_white(clothes_hvs, hsva_noise, 0.2, 0.2)
        socks_hsva = randomise_to_black_and_white(clothes_hvs, hsva_noise, 0.25, 0.25)
        shoes_hsva = randomise_to_black_and_white(clothes_hvs, hsva_noise, 0.2, 0.2)
        body_hsv = np.random.uniform(
            [args.min_player_skin_hue, args.min_player_skin_sat, args.min_player_skin_val],
            [args.max_player_skin_hue, args.max_player_skin_sat, args.max_player_skin_val],
        )
        bpy.data.materials[f"player_shirt_{team_name}"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].color = shirt_hsva
        bpy.data.materials[f"player_shorts_{team_name}"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].color = shorts_hsva
        bpy.data.materials[f"player_socks_{team_name}"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].color = socks_hsva
        bpy.data.materials[f"player_shoes_{team_name}"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].color = shoes_hsva
        bpy.data.materials[f"player_body_{team_name}"].node_tree.nodes["Hue Saturation Value"].inputs[0].default_value = body_hsv[0]
        bpy.data.materials[f"player_body_{team_name}"].node_tree.nodes["Hue Saturation Value"].inputs[1].default_value = body_hsv[1]
        bpy.data.materials[f"player_body_{team_name}"].node_tree.nodes["Hue Saturation Value"].inputs[2].default_value = body_hsv[2]

        created_collection_names = []
        num_players = np.random.randint(args.min_players, args.max_players+1)
        for i in range(num_players):
            new_collection_name = f"{collection_name}_copy_{str(i).zfill(2)}"
            animation_frame = np.random.randint(0, self.MAX_PLAYER_ANIMATION_FRAMES+1)
            copy_player_and_freeze_at_frame(animation_frame, collection_name, new_collection_name)
            select_all_meshes_in_collection(new_collection_name)
            scaling_factor = np.random.uniform(args.min_player_scale, args.max_player_scale, 3)
            bpy.ops.transform.resize(value=scaling_factor, orient_type="GLOBAL",
                                     orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type="GLOBAL",
                                     constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False,
                                     proportional_edit_falloff="SMOOTH", proportional_size=1,
                                     use_proportional_connected=False, use_proportional_projected=False, snap=False,
                                     snap_elements={"INCREMENT"}, use_snap_project=False, snap_target="CLOSEST",
                                     use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True,
                                     use_snap_selectable=False, release_confirm=True)
            translate_kwargs = dict(
                orient_type="GLOBAL",
                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type="GLOBAL",
                constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False,
                proportional_edit_falloff="SMOOTH", proportional_size=1,
                use_proportional_connected=False, use_proportional_projected=False, snap=False,
                snap_elements={"INCREMENT"}, use_snap_project=False, snap_target="CLOSEST",
                use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True,
                use_snap_selectable=False, release_confirm=True
            )

            rot = np.random.uniform(
                [-5 * np.pi / 180.0, -5 * np.pi / 180.0, -np.pi],
                [5 * np.pi / 180.0, 5 * np.pi / 180.0, np.pi]
            )
            rot_kwargs = dict(
                orient_type="GLOBAL",
                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type="GLOBAL",
                constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False,
                proportional_edit_falloff="SMOOTH", proportional_size=1,
                use_proportional_connected=False, use_proportional_projected=False, snap=False,
                snap_elements={"INCREMENT"}, use_snap_project=False, snap_target="CLOSEST",
                use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True,
                use_snap_selectable=False, release_confirm=True
            )
            bpy.ops.transform.rotate(value=rot[0], orient_axis="X", **rot_kwargs)
            bpy.ops.transform.rotate(value=rot[1], orient_axis="Y", **rot_kwargs)
            bpy.ops.transform.rotate(value=rot[2], orient_axis="Z", **rot_kwargs)

            shoe_mesh_name = next(obj.name for obj in bpy.data.collections[new_collection_name].all_objects if "Shoes" in obj.name)
            wanted_position = np.random.uniform(
                [-0.5*self.FIELD_X_BOUNDS, -0.5*self.FIELD_Y_BOUNDS],
                [0.5*self.FIELD_X_BOUNDS, 0.5*self.FIELD_Y_BOUNDS],
            )
            current_position = np.array(bpy.data.objects[shoe_mesh_name].matrix_world.translation[:2])
            translation = wanted_position - current_position
            print(f"{team_name}[{i}]: ({shoe_mesh_name}): player is at {current_position}, wanted at {wanted_position}: {translation}")
            bpy.ops.transform.translate(value=(translation[0], 0, 0), orient_axis_ortho="X", **translate_kwargs)
            bpy.ops.transform.translate(value=(0, translation[1], 0), orient_axis_ortho="Y", **translate_kwargs)

            created_collection_names.append(new_collection_name)
        return created_collection_names

    def _sample_camera_looking_at_ball(self, args: GenerationSettings, ball_loc: np.ndarray):
        camera_loc = np.random.uniform(
            [-self.FIELD_X_BOUNDS * 0.5, -self.FIELD_Y_BOUNDS * 0.5, args.camera_min_z],
            [self.FIELD_X_BOUNDS * 0.5, -self.FIELD_Y_BOUNDS * 0.5, args.camera_max_z],
        )
        camera_ball = ball_loc - camera_loc
        rot_quat = mathutils.Vector(camera_ball).to_track_quat("-Z", "Y")
        rot_euler = rot_quat.to_euler()
        cam_angle_noise = 5 * np.pi / 180
        rot_euler += np.random.uniform(
            [-cam_angle_noise, -cam_angle_noise, -cam_angle_noise],
            [cam_angle_noise, cam_angle_noise, cam_angle_noise],
        )
        bpy.data.objects["Camera"].location = camera_loc
        bpy.data.objects["Camera"].rotation_euler = rot_euler

    def _save_img_and_ann(self, context, args: GenerationSettings, file_stem: str):
        file_path = (Path(args.save_dir) / args.save_format.replace(REPLACE_TOKEN, file_stem)).resolve().absolute()
        ann_path = file_path.parent / f"{file_path.stem}.json"

        def bbox_to_dict(b):
            return {"x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1], }

        ann = {
            "image": {
                "filename": file_path.name,
                "height": context.scene.render.resolution_x,
                "width": context.scene.render.resolution_y,
            },
            "annotations": []
        }
        print(f"saving images and annotations to {file_path}, {ann_path}")
        context.scene.render.filepath = str(file_path)
        bpy.ops.render.render(animation=False, write_still=True)

        ball_found, ball_bbox = camera_view_bounds_2d(context.scene, bpy.data.objects["Camera"], [bpy.data.objects["Soccer Ball"]])
        if ball_found:
            print(f"found ball {ball_bbox}")
            ann["annotations"].append({
                "bounding_box": bbox_to_dict(ball_bbox),
                "name": "ball",
            })
        keypoint_bboxes = {}
        for collection in bpy.data.collections:
            if "player_mesh_team_" in collection.name:
                found, bbox = camera_view_bounds_2d(
                    context.scene,
                    bpy.data.objects["Camera"],
                    list(collection.all_objects),
                )
                if found:
                    print(f"found player {collection.name}: {bbox}")
                    ann["annotations"].append({
                        "bounding_box": bbox_to_dict(bbox),
                        "name": "human",
                    })
            for obj in collection.all_objects:
                name = obj.name
                if name.startswith(self.KEYPOINT_PREFIX):
                    dot_idx = name.find(".")
                    if dot_idx != -1:
                        kp_name = name[len(self.KEYPOINT_PREFIX): dot_idx]
                    else:
                        kp_name = name[len(self.KEYPOINT_PREFIX):]
                    found, kp_bbox = camera_view_bounds_2d(
                        context.scene,
                        bpy.data.objects["Camera"],
                        [obj],
                    )
                    if found:
                        keypoint_bboxes[name] = kp_bbox
                        print(f"found keypoint {collection.name}/{kp_name}: {kp_bbox}")
                        ann["annotations"].append({
                            "bounding_box": bbox_to_dict(kp_bbox),
                            "name": kp_name,
                        })
        ann_path.write_text(json.dumps(ann, indent=4))
        print(f"saved images and annotations to {file_path}, {ann_path}")


class GenerateDataPanel(bpy.types.Panel):
    bl_label = "Generate Data"
    bl_idname = "OBJECT_GENERATE_DATA2"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GenerateData"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.data_gen_settings

        row = layout.row()
        row.label(text="Sampling Parameters", icon="RESTRICT_RENDER_OFF")
        for name in [
            "min_players", "max_players",
            "min_player_hue", "max_player_hue",
            "min_player_sat", "max_player_sat",
            "min_player_val", "max_player_val",
            "min_player_skin_hue", "max_player_skin_hue",
            "min_player_skin_sat", "max_player_skin_sat",
            "min_player_skin_val", "max_player_skin_val",
            "min_player_scale", "max_player_scale",
            "min_light_intensity", "max_light_intensity",
            "min_light_cone_angle_deg", "max_light_cone_angle_deg",
            "min_field_hue", "max_field_hue",
            "min_field_sat", "max_field_sat",
            "min_field_val", "max_field_val",
            "shade_chance", "shade_min", "shade_max",
            "camera_min_z", "camera_max_z",
            "SEP",
            "save_dir", "save_format",
            "images_per_config", "n_images",
        ]:
            if name == "SEP":
                layout.separator()
                continue
            row = layout.row()
            row.prop(settings, name)
        
        layout.separator()
        row = layout.row()
        row.operator("render.take_rendered_image")


settings_classes = [
    GenerationSettings,
]
classes = [
    GenerateDataPanel,
    TakeRenderedImages,
]


def register():
    for c in classes + settings_classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.data_gen_settings = bpy.props.PointerProperty(type=GenerationSettings)


def unregister():
    for c in classes + settings_classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.data_gen_settings


if __name__ == "__main__":
    register()
