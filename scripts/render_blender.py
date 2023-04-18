import glob
import math
import os
from typing import List

import bpy
import numpy as np


def render_obj(obj_path: str, num_views: int, output_folder: str, depth_scale: float = 0.1, scale: int = 1, remove_doubles: bool = True,
               edge_split: bool = False, color_depth: int = 8, resolution: int = 600, engine: str = 'BLENDER_EEVEE'):
    """
    Generates rgbd images (rgb + depth) for a given mesh (.obj file path)
    source: https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
    :param obj_path: Path to the obj file to be rendered.
    :param texture_path: Path to texture image.
    :param num_views: number of views to be rendered
    :param output_folder: The path the output will be dumped to.
    :param depth_scale: Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result
    :param scale: Scaling factor applied to model. Depends on size of mesh.
    :param remove_doubles: Remove double vertices to improve mesh quality.
    :param edge_split: Adds edge split filter.
    :param color_depth: Number of bit per channel used for output. Either 8 or 16.
    :param resolution: Resolution of the images.
    :param engine: Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...
    """
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render
    format_img = "PNG"

    render.engine = engine
    render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = str(color_depth)  # ('8', '16')
    render.image_settings.file_format = format_img  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.film_transparent = True

    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = '/'
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = format_img
    depth_file_output.format.color_depth = str(color_depth)

    depth_file_output.format.color_mode = "BW"

    # Remap as other types can not represent the full range of depth.
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.size = [depth_scale]
    map.use_min = True
    map.min = [0]
    map.use_max = True
    map.max = [255]

    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

    # Delete default cube
    objs_to_delete = [i for i in bpy.data.objects if i.name not in ['Camera', 'Light']]
    for obj_to_delete in objs_to_delete:
        bpy.data.objects.remove(obj_to_delete, do_unlink=True)

    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.import_scene.obj(filepath=obj_path)

    obj = bpy.context.selected_objects[0]
    context.view_layer.objects.active = obj

    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes['Principled BSDF']
        node.inputs['Specular'].default_value = 0.05

    if scale != 1:
        bpy.ops.transform.resize(value=(scale, scale, scale))
        bpy.ops.object.transform_apply(scale=True)
    if remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Set object IDs
    obj.pass_index = 1

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 10.0

    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 0.015
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180

    # Place camera
    cam = scene.objects['Camera']
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    stepsize = 360.0 / num_views
    rotation_mode = 'XYZ'

    path_model, model_id_1 = os.path.split(os.path.split(os.path.split(obj_path)[0])[0])
    model_id_0 = os.path.split(path_model)[1]
    fp = os.path.abspath(output_folder)

    sc = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * sc # px
    height = scene.render.resolution_y * sc  # px
    camdata = scene.camera.data
    focal = camdata.lens  # mm
    sensor_width = camdata.sensor_width  # mm
    sensor_height = camdata.sensor_height  # mm
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if (camdata.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = width / sensor_width / pixel_aspect_ratio
        s_v = height / sensor_height
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = width / sensor_width
        s_v = height * pixel_aspect_ratio / sensor_height

    # parameters of intrinsic calibration matrix K
    alpha_u = focal * s_u
    alpha_v = focal * s_v
    u_0 = width / 2
    v_0 = height / 2
    skew = 0  # only use rectangular pixels

    K = np.array([
        [alpha_u, skew, u_0],
        [0, alpha_v, v_0],
        [0, 0, 1]
    ], dtype=np.float32)

    for i in range(0, num_views):
        print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

        file_name = f'{model_id_1}_{int(i * stepsize):03d}'
        scene.render.filepath = os.path.join(fp, model_id_0, 'rgb', f'{file_name}0001')
        depth_file_output.file_slots[0].path = os.path.join(fp, model_id_0, 'depth', file_name)

        bpy.ops.render.render(write_still=True)  # render still

        cam_empty.rotation_euler[2] += math.radians(stepsize)

    np.save(os.path.join(fp, 'camera.npy'), K)


def create_datasets(input_models_path: str, num_views_per_obj: int, output_path: str, split: List, **kwargs):
    """
    Creates train, validation and test datasets
    :param input_models_path: path containing all models from shapenet
    :param num_views_per_obj: number of views (images) for a given object
    :param output_path: path where the whole dataset should be saved
    :param split: list of ratios between train, val and test, e.g. [0.7, 0.1, 0.2]
    :param kwargs: additional parameters, passed to function render_obj
    """
    #TODO: think about reasonable splitting criterion
    np.random.seed(23)
    split = np.array(split)
    all_models = glob.glob(input_models_path + '/*')
    num_all_models = len(all_models)
    all_models = np.random.permutation(all_models)

    if np.sum(split) != 1:
        split /= np.sum(split)

    train_size = int(num_all_models * split[0])
    val_size = int(num_all_models * split[1])

    train_models = all_models[:train_size]
    val_models = all_models[train_size: train_size + val_size]
    test_models = all_models[train_size + val_size:]

    train_models = [glob.glob(t + '/*/*/*.obj') for t in train_models]
    train_models = sum(train_models, [])
    val_models = [glob.glob(t + '/*/*/*.obj') for t in val_models]
    val_models = sum(val_models, [])
    test_models = [glob.glob(t + '/*/*/*.obj') for t in test_models]
    test_models = sum(test_models, [])

    for entry in [(train_models, 'train'), (val_models, 'val'), (test_models, 'test')]:
        objs, name = entry
        output_path_name = os.path.join(output_path, name)
        for obj in objs:
            print(obj)
            render_obj(obj, num_views_per_obj, output_path_name, **kwargs)


if __name__ == '__main__':
    input_path = '../data/shape_net_3_classes/'
    depth_scale = 0.3
    num_views = 30
    output_path = '../data/images/'
    train_val_test_split = [0.7, 0.1, 0.2]
    create_datasets(input_path, num_views, output_path, train_val_test_split, depth_scale=depth_scale)
