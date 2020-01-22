"""
Generates 2D and 3D renderings of crystal cells and phase plots.
"""

from . import crystal
import itertools
import numpy
import scipy
import os
import tempfile

def _cell_process_kwargs(kwargs):
    """
    Dispatches keyword arguments as necessary to helper functions.

    Parameters
    ----------
    kwargs : dict(str, object)
        The arguments provided to a frontend rendering function.

    Returns
    -------
    cell_kwargs, sphere_kwargs, draw_kwargs : dict(str, object), dict(str, object), dict(str, object)
        The arguments to be dispatched to :py:func:`_cell`, the arguments to
        be dispatched to :py:func:`_sphere`, and the arguments to be dispatched
        to drawing functions, respectively.
    """

    # Specify the kwargs to be pulled out
    cell_kwargs_allowed, sphere_kwargs_allowed = \
        {"supercell_box", "cell_boxes", "partial", "partial_tolerance", "at_contact", "all_types"}, {"resolution"}
    cell_kwargs, sphere_kwargs = {}, {}
    draw_kwargs = dict(kwargs)

    # Pull out the kwargs
    for kwarg in kwargs:
        if kwarg in cell_kwargs_allowed:
            cell_kwargs[kwarg] = draw_kwargs[kwarg]
            del draw_kwargs[kwarg]
        if kwarg in sphere_kwargs_allowed:
            sphere_kwargs[kwarg] = draw_kwargs[kwarg]
            del draw_kwargs[kwarg]

    return cell_kwargs, sphere_kwargs, draw_kwargs

def _cell(cell, repeats, radii, supercell_box=False, cell_boxes=False, partial=False, partial_tolerance=1e-6, at_contact=True, all_types=None):
    """
    Generates data to make a rendering of a crystal cell.  The cell is rendered at
    closest contact based on the radii specified.

    Parameters
    ----------
    cell : paccs.crystal.Cell
        The cell to render.
    repeats : tuple(int)
        The number of repeats in each dimension.
    radii : tuple(float)
        Radii of the atom types.
    supercell_box : bool
        Whether or not to draw a box around the entire supercell.
    cell_boxes : bool
        Whether or not to draw a box around individual cells within the supercell.
    partial : bool
        Whether or not to generate additional partial periodic images.
    partial_tolerance : float
        See :py:func:`paccs.crystal.CellTools.tile`.
    at_contact : bool
        Whether or not to draw the cell at nearest neighbor contact determined by radii.
    all_types : list(str)
        Names of all atoms to be considered allowable.  This allows
        consistent coloring across many different cells which may have
        certain types missing from certain cells.

    Raises
    ------
    NotImplementedError
        The cell is not 2-dimensional or 3-dimensional.

    Returns
    -------
    coordinates, colors, sizes, types : numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Coordinates in 3D of the atoms to be rendered, values corresponding to the
        colors of the atoms to be rendered, radii of the atoms to be rendered, and
        types of the atoms, respectively.
    """

    # Check dimensionality of cell
    if cell.dimensions not in {2, 3}:
        raise NotImplementedError("only renderings of 2D or 3D cells may be made.")

    # Get coordinates directly from supercell
    supercell = crystal.CellTools.tile(cell, repeats, \
        partial_radii=radii if partial else None, partial_tolerance=partial_tolerance)
    coordinates = numpy.concatenate(supercell.atom_lists)

    # Color atoms by type (Mayavi autoscales based on the selected colormap)
    if (all_types == None):
        colors = numpy.array([type_index / (supercell.atom_types - int(supercell.atom_types > 1)) \
            for type_index in range(supercell.atom_types) \
            for atom_index in range(supercell.atom_count(type_index))])
    else:
        if (not numpy.all([not isinstance(k,int) for k in all_types])): raise Exception("all_types must be given as a list of atom names (strings)")
        global_types = {sorted(all_types)[i]:i for i in range(len(all_types))}
        try:
            colors = numpy.array([global_types[supercell.name(type_index)] / (len(all_types) - int(len(all_types) > 1)) \
                for type_index in range(supercell.atom_types) \
                for atom_index in range(supercell.atom_count(type_index))])
        except Exception as e:
            raise Exception("unable to cell atoms to map globally allowed atom types : {}".format(e))

    # Size atoms based on radius and contact information (use cell contact information, not supercell)
    sizes = numpy.array(radii) * (cell.scale_factor(radii) if at_contact else 1)
    sizes = numpy.array([sizes[type_index] \
        for type_index in range(supercell.atom_types) \
        for atom_index in range(supercell.atom_count(type_index))])

    # Return the types directly in addition to within the color information
    types = numpy.array([type_index \
        for type_index in range(supercell.atom_types) \
        for atom_index in range(supercell.atom_count(type_index))])

    # If the rendering is 2D, set all z-coordinates to 0
    if cell.dimensions != 3:
        coordinates = numpy.concatenate((coordinates, numpy.zeros((coordinates.shape[0], 1))), axis=1)

    # Render boxes
    boxes = []
    if supercell_box:
        boxes.append((numpy.zeros(cell.dimensions), supercell.vectors))
    if cell_boxes:
        for cell_box in itertools.product(*(range(repeat) for repeat in repeats)):
            boxes.append((numpy.dot(cell.vectors.T, cell_box).T, cell.vectors))

    return coordinates, colors, sizes, types, boxes

def _sphere(coordinates, radius=1, resolution=(16, 16), scale_factor=0.5):
    """
    Generates Cartesian coordinates for a sphere.

    Parameters
    ----------
    coordinates : numpy.ndarray
        The coordinates of the center of the sphere.
    radius : float
        The radius of the sphere.
    resolution : tuple(float)
        The resolution in the theta and phi directions, respectively.
    scale_factor : float
        Controls the degree to which the sphere is scaled up to compensate for
        reduced resolution (prevents gaps from appearing between spheres with
        tangent contact points).

    Returns
    -------
    x, y, z : numpy.ndarray, numpy.ndarray, numpy.ndarray
        Cartesian coordinates on the surface of the sphere.
    """

    # Generate grid in spherical coordinates
    theta = numpy.linspace(0, 2 * numpy.pi, resolution[0])
    phi = numpy.linspace(0, numpy.pi, resolution[1])
    theta, phi = numpy.meshgrid(theta, phi)

    # Scale spheres up so that no space appears between them at low resolutions
    radius *= (2 - numpy.cos(2 * numpy.pi * scale_factor / min(resolution)))

    # Generate grid in Cartesian coordinates
    x = coordinates[0] + (radius * numpy.cos(theta) * numpy.sin(phi))
    y = coordinates[1] + (radius * numpy.sin(theta) * numpy.sin(phi))
    z = coordinates[2] + (radius * numpy.cos(phi))
    return x, y, z

def _mayavi_cast_ray(scene, x, y):
    """
    Determines the parametrization of a ray in a scene based on display coordinates.

    Parameters
    ----------
    scene : mayavi.core.api.Scene
        Mayavi scene with information on the viewport and camera.
    x : float
        The x-component of the display coordinates.
    y : float
        The y-component of the display coordinates.

    Returns
    -------
    origin, direction : numpy.ndarray, numpy.ndarray
        A point representing the origin of the ray, and a normalized vector pointing
        along the ray.
    """

    # Normalize display coordinates
    width, height = scene.get_size()
    normalized_coordinates = numpy.array([(2 * x / width) - 1, (2 * y / height) - 1])

    if scene.camera.parallel_projection: # Orthographic rendering mode
        # Get camera coordinate system
        out_direction = scene.camera.direction_of_projection
        up_direction = scene.camera.view_up
        right_direction = numpy.cross(out_direction, up_direction)

        # Calculate origin and direction of ray
        return scene.camera.position + \
            (((normalized_coordinates[0] * right_direction * width / height) + \
            (normalized_coordinates[1] * up_direction)) * \
            scene.camera.parallel_scale), out_direction

    else: # Perspective rendering mode
        # Get the transformation matrix
        projection_matrix = scipy.linalg.inv(scene.camera.get_composite_projection_transform_matrix( \
            width / height, *scene.camera.clipping_range).to_array())

        # Do the transformation itself
        screen_point = numpy.concatenate([normalized_coordinates, numpy.ones(2)])
        world_point = numpy.dot(projection_matrix, screen_point)
        vector = (world_point[:-1] / world_point[-1]) - scene.camera.position

        return scene.camera.position, vector / numpy.linalg.norm(vector)

def cell_mayavi(cell, repeats, radii, **kwargs):
    """
    Generates a 2D or 3D rendering of a crystal cell using Mayavi.
    The cell is rendered at closest contact based on the radii specified.

    Parameters
    ----------
    cell : paccs.crystal.Cell
        The cell to render.
    repeats : tuple(int)
        The number of repeats in each dimension.
    radii : tuple(float)
        Radii of the atom types by cell's internal indexing.
    kwargs : dict(str, object)
        Additional arguments to pass to various helper functions.

        save_to : str
            The rendered image to be saved to this file.
        all_types : list(str)
            Names of all atoms to be considered allowable.  This allows
            consistent coloring across many different cells which may have
            certain types missing from certain cells. If cell does not contain
            these, an Exception will be thrown.
        at_contact : bool
            Default is to draw atoms at contact determined by radii, but if
            this is False, atoms will be rendered at their absolute scale.
        magnification : float
            The magnification is the scaling between the pixels on the screen, and the pixels in the file saved, if ``save_to`` is given.
            Defaults to 1, see :py:func:`mayavi.mlab.savefig()` for more details.

    Raises
    ------
    NotImplementedError
        The cell is not 2-dimensional or 3-dimensional.
    ImportError
        Mayavi is unavailable on the system.
    Exception
        Cannot find cell's internally named atoms in all_types, if specified.
    """

    from mayavi import mlab
    mlab.figure(bgcolor=(1,1,1)) # White background

    save_to = None
    if ("save_to" in kwargs):
        save_to = kwargs['save_to']
        del kwargs['save_to']

    magnification = 1.0
    if ("magnification" in kwargs):
        magnification = kwargs['magnification']
        del kwargs['magnification']

    cell_kwargs, sphere_kwargs, draw_kwargs = _cell_process_kwargs(kwargs)

    # Calculate points' properties
    coordinates, colors, sizes, types, boxes = _cell(cell, repeats, radii, **cell_kwargs)

    # Draw spheres
    for sphere_index in range(len(types)):
        x, y, z = _sphere(coordinates[sphere_index], sizes[sphere_index], **sphere_kwargs)
        surface = mlab.mesh(x, y, z, scalars=colors[sphere_index] * numpy.ones_like(z), \
            vmin=0, vmax=1, **draw_kwargs)

    # Draw boxes
    box_options = dict(color=(0, 0, 0), opacity=0.5, representation="wireframe", tube_radius=None)
    for box in boxes:
        if cell.dimensions == 2:
            mlab.plot3d(box[0][0] + numpy.array([0, box[1][0, 0], box[1][0, 0] + box[1][1, 0], box[1][1, 0], 0]), \
                box[0][1] + numpy.array([0, box[1][0, 1], box[1][0, 1] + box[1][1, 1], box[1][1, 1], 0]), \
                numpy.zeros((5)), **box_options)
        else:
            mlab.plot3d(box[0][0] + numpy.array([0, box[1][0, 0], box[1][0, 0] + box[1][1, 0], box[1][1, 0], 0]), \
                box[0][1] + numpy.array([0, box[1][0, 1], box[1][0, 1] + box[1][1, 1], box[1][1, 1], 0]), \
                box[0][2] + numpy.array([0, box[1][0, 2], box[1][0, 2] + box[1][1, 2], box[1][1, 2], 0]), **box_options)
            mlab.plot3d(box[0][0] + box[1][2, 0] + numpy.array([0, box[1][0, 0], box[1][0, 0] + box[1][1, 0], box[1][1, 0], 0]), \
                box[0][1] + box[1][2, 1] + numpy.array([0, box[1][0, 1], box[1][0, 1] + box[1][1, 1], box[1][1, 1], 0]), \
                box[0][2] + box[1][2, 2] + numpy.array([0, box[1][0, 2], box[1][0, 2] + box[1][1, 2], box[1][1, 2], 0]), **box_options)
            mlab.plot3d(box[0][0] + numpy.array([0, box[1][1, 0], box[1][1, 0] + box[1][2, 0], box[1][2, 0], 0]), \
                box[0][1] + numpy.array([0, box[1][1, 1], box[1][1, 1] + box[1][2, 1], box[1][2, 1], 0]), \
                box[0][2] + numpy.array([0, box[1][1, 2], box[1][1, 2] + box[1][2, 2], box[1][2, 2], 0]), **box_options)
            mlab.plot3d(box[0][0] + box[1][0, 0] + numpy.array([0, box[1][1, 0], box[1][1, 0] + box[1][2, 0], box[1][2, 0], 0]), \
                box[0][1] + box[1][0, 1] + numpy.array([0, box[1][1, 1], box[1][1, 1] + box[1][2, 1], box[1][2, 1], 0]), \
                box[0][2] + box[1][0, 2] + numpy.array([0, box[1][1, 2], box[1][1, 2] + box[1][2, 2], box[1][2, 2], 0]), **box_options)
            mlab.plot3d(box[0][0] + numpy.array([0, box[1][2, 0], box[1][2, 0] + box[1][0, 0], box[1][0, 0], 0]), \
                box[0][1] + numpy.array([0, box[1][2, 1], box[1][2, 1] + box[1][0, 1], box[1][0, 1], 0]), \
                box[0][2] + numpy.array([0, box[1][2, 2], box[1][2, 2] + box[1][0, 2], box[1][0, 2], 0]), **box_options)
            mlab.plot3d(box[0][0] + box[1][1, 0] + numpy.array([0, box[1][2, 0], box[1][2, 0] + box[1][0, 0], box[1][0, 0], 0]), \
                box[0][1] + box[1][1, 1] + numpy.array([0, box[1][2, 1], box[1][2, 1] + box[1][0, 1], box[1][0, 1], 0]), \
                box[0][2] + box[1][1, 2] + numpy.array([0, box[1][2, 2], box[1][2, 2] + box[1][0, 2], box[1][0, 2], 0]), **box_options)

    # Create an object used to highlight selected atoms
    highlight = mlab.outline()
    highlight.visible = False

    # If the rendering is 2D, look from above with an orthographic viewpoint
    if cell.dimensions != 3:
        mlab.gcf().scene.parallel_projection = True
        mlab.view(0, 0)

    # Called to update selection
    def update_selection(atom_index):
        if atom_index == -1:
            highlight.visible = False
        else:
            box_size = sizes[atom_index] / numpy.sqrt(2)
            highlight.bounds = ( \
                coordinates[atom_index, 0] - box_size, coordinates[atom_index, 0] + box_size, \
                coordinates[atom_index, 1] - box_size, coordinates[atom_index, 1] + box_size, \
                coordinates[atom_index, 2] - box_size, coordinates[atom_index, 2] + box_size)
            highlight.visible = True
            print("{} atom at ({:.6f}, {:.6f}, {:.6f})".format(cell.name(types[atom_index]), *coordinates[atom_index]))

    # Mouse selection
    def picker_callback(picker):
        mouse_x, mouse_y, mouse_z = picker.selection_point
        origin, direction = _mayavi_cast_ray(mlab.gcf().scene, mouse_x, mouse_y)

        # Check all atoms for collision
        minimum_atom, minimum_distance = -1, numpy.inf
        for atom_index in range(len(sizes)):
            # Calculate two parameters used to determine the collision distance
            alpha = numpy.dot(direction, origin - coordinates[atom_index])
            beta = (numpy.linalg.norm(origin - coordinates[atom_index]) ** 2) - (sizes[atom_index] ** 2)

            # If the following discriminant is negative, the atom is missed
            discriminant = (alpha ** 2) - beta
            if discriminant < 0:
                continue

            # A line passes through the atom, make sure that it is not behind the camera
            if alpha > 0:
                continue

            # The atom is hit, get the distance
            distance = -alpha - numpy.sqrt(discriminant)
            if distance < minimum_distance:
                minimum_atom, minimum_distance = atom_index, distance

        update_selection(minimum_atom)

    picker = mlab.gcf().on_mouse_pick(picker_callback)
    if (save_to is None):
        mlab.show()
    else:
        mlab.savefig(save_to, figure=mlab.gcf(), magnification=magnification)
        mlab.close()

def cell_plotly(cell, repeats, radii, **kwargs):
    """
    Generates a 2D or 3D rendering of a crystal cell using Plotly.
    The cell is rendered at closest contact based on the radii specified.

    Parameters
    ----------
    cell : paccs.crystal.Cell
        The cell to render.
    repeats : tuple(int)
        The number of repeats in each dimension.
    radii : tuple(int)
        Radii of the atom types.
    kwargs : dict(str, object)
        Additional arguments to pass to various helper functions.

    Raises
    ------
    NotImplementedError
        The cell is not 2-dimensional or 3-dimensional, or boxes were requested.
    ImportError
        Plotly is unavailable on the system.
    """

    # Gather data
    import plotly
    cell_kwargs, sphere_kwargs, draw_kwargs = _cell_process_kwargs(kwargs)
    coordinates, colors, sizes, types, boxes = _cell(cell, repeats, radii, **cell_kwargs)
    if boxes:
        raise NotImplementedError("boxes are not currently supported by this renderer")

    # Build geometry
    data = []
    for sphere_index in range(len(types)):
        x, y, z = _sphere(coordinates[sphere_index], sizes[sphere_index], **sphere_kwargs)
        data.append(plotly.graph_objs.Surface(x=x, y=y, z=z, surfacecolor=colors[sphere_index] * numpy.ones_like(z), \
            cmin=0, cmax=1, showscale=False, **draw_kwargs))

    # Render
    figure = plotly.graph_objs.Figure(data=data, layout=plotly.graph_objs.Layout())
    handle, filename = tempfile.mkstemp(".html", "paccs-")
    os.close(handle)
    plotly.offline.offline.plot(figure, filename=filename)
