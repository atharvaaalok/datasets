import gmsh
import numpy as np


def mesh_func(
    X,
    y_first_layer,
    growth_ratio,
    total_BL_thickness,
    mesh_size_at_airfoil,
    mesh_size_at_farfield,
    farfield_factor,
    fan_points_coordinates_list = None,
    fan_count = 10,
    mesh_file_path_list = None,
    visualize = False,
    verbose = False,
    model_name = 'airfoil',
):
    
    # Initialize gmsh and add a model
    gmsh.initialize()
    if not verbose:
        # Don't print outputs to stdout but keep errors and warnings
        gmsh.option.setNumber("General.Verbosity", 1)
    
    gmsh.model.add(model_name)


    # Initialize list of points on shape and add points
    points_on_shape = []
    fan_points_list = []
    

    # Add points on shape and to the fan list
    total_points_on_shape = X.shape[0] - 1
    point_tag = 1
    for i in range(total_points_on_shape):
        x, y = X[i, 0].item(), X[i, 1].item()

        points_on_shape.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size_at_airfoil, point_tag))

        if (x, y) in fan_points_coordinates_list:
            fan_points_list.append(point_tag)

        point_tag += 1
    

    # Initialize list of lines on shape and add lines
    lines_on_shape = []

    total_lines_on_shape = total_points_on_shape
    line_tag = 1
    for i in range(total_lines_on_shape):
        if i == total_lines_on_shape - 1:
            pt1, pt2 = points_on_shape[i], points_on_shape[0]
        else:
            pt1, pt2 = points_on_shape[i], points_on_shape[i + 1]
        
        lines_on_shape.append(gmsh.model.geo.addLine(pt1, pt2, line_tag))

        line_tag += 1
    

    # Create a curve loop for the shape
    curve_loop_tag = 1
    curve_loop_shape = gmsh.model.geo.addCurveLoop(lines_on_shape, curve_loop_tag)
    curve_loop_tag += 1


    # Create far field boundary
    xmin, xmax = -1 * farfield_factor, 1 * farfield_factor
    ymin, ymax = -1 * farfield_factor, 1 * farfield_factor


    # Add boundary points
    points_on_boundary = []
    points_on_boundary.append(gmsh.model.geo.addPoint(xmin, ymin, 0, mesh_size_at_farfield, point_tag))
    point_tag += 1
    points_on_boundary.append(gmsh.model.geo.addPoint(xmin, ymax, 0, mesh_size_at_farfield, point_tag))
    point_tag += 1
    points_on_boundary.append(gmsh.model.geo.addPoint(xmax, ymax, 0, mesh_size_at_farfield, point_tag))
    point_tag += 1
    points_on_boundary.append(gmsh.model.geo.addPoint(xmax, ymin, 0, mesh_size_at_farfield, point_tag))
    point_tag += 1


    # Add boundary lines
    lines_on_boundary = []
    lines_on_boundary.append(gmsh.model.geo.addLine(points_on_boundary[0], points_on_boundary[1], line_tag))
    line_tag += 1
    lines_on_boundary.append(gmsh.model.geo.addLine(points_on_boundary[1], points_on_boundary[2], line_tag))
    line_tag += 1
    lines_on_boundary.append(gmsh.model.geo.addLine(points_on_boundary[2], points_on_boundary[3], line_tag))
    line_tag += 1
    lines_on_boundary.append(gmsh.model.geo.addLine(points_on_boundary[3], points_on_boundary[0], line_tag))
    line_tag += 1


    # Create a curve loop for the boundary
    curve_loop_boundary = gmsh.model.geo.addCurveLoop(lines_on_boundary, curve_loop_tag)
    curve_loop_tag += 1


    # Create a plane surface for the domain
    plane_surface_tag = 1
    plane_surface = gmsh.model.geo.addPlaneSurface([curve_loop_boundary, curve_loop_shape], plane_surface_tag)
    plane_surface_tag += 1


    gmsh.model.geo.synchronize()

    airfoil_curves = lines_on_shape
    farfield_curves = lines_on_boundary

    gmsh.model.geo.synchronize()


    # Add physical groups
    gmsh.model.addPhysicalGroup(1, airfoil_curves, name = 'airfoil')
    gmsh.model.addPhysicalGroup(1, farfield_curves, name = 'farfield')
    gmsh.model.addPhysicalGroup(2, [plane_surface], name = 'plane_surface')

    # Save tags of entities in physical groups in a dictionary
    physical_groups = {
        'airfoil_curves': airfoil_curves,
        'farfield_curves': farfield_curves,
        'plane_surface': plane_surface
    }


    # Add mesh field
    f = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(f, 'CurvesList', lines_on_shape)
    gmsh.model.mesh.field.setNumber(f, 'Size', y_first_layer)
    gmsh.model.mesh.field.setNumber(f, 'Ratio', growth_ratio)
    gmsh.model.mesh.field.setNumber(f, 'Quads', 1)
    gmsh.model.mesh.field.setNumber(f, 'Thickness', total_BL_thickness)

    # Create a fan at the trailing edge
    gmsh.option.setNumber('Mesh.BoundaryLayerFanElements', fan_count)
    if len(fan_points_list) != 0:
        gmsh.model.mesh.field.setNumbers(f, 'FanPointsList', fan_points_list)

    gmsh.model.mesh.field.setAsBoundaryLayer(f)


    # Generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)


    # Get total nodes and elements in the mesh
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    num_nodes = len(nodeTags)
    _, elementTags, _ = gmsh.model.mesh.getElements()
    num_elements = sum(len(tags) for tags in elementTags)
    if verbose:
        print(f"{num_nodes} nodes, {num_elements} elements")
    

    # Create a dictionary that holds mesh details and will be returned from the function
    mesh_details = {'num_nodes': num_nodes, 'num_elements': num_elements,
                    'physical_groups': physical_groups}


    # Save mesh to file if file path is provided
    if mesh_file_path_list is not None:
        # Save a mesh file for every format specified
        for filename in mesh_file_path_list:
            gmsh.write(filename)

    if visualize:
        gmsh.fltk.run()

    gmsh.finalize()

    return mesh_details