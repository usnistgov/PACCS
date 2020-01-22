Wallpaper groups
================

The generation routines within paccs use wallpaper groups to generate
candidate cells in two dimensions.  This document describes some of the features
of the wallpaper groups as well as how they are represented within paccs.

The groups themselves
---------------------

There are exactly 17 wallpaper groups, summarized in the following table and
displayed in the graphic below.  The table indicates, respectively:

1. The index (starting from 1) of each group as used by paccs.

2. The Hermann-Mauguin identifier for the group.

3. The orbifold notation identifier for the group.

4. The cosine of the angle between the vectors of the fundamental domain (aka "tile") 
   which the wallpaper group uses to tile space, if such a constraint exists.
   Note that there is not a unique choice for these vector pairs; this simply reflects
   our convention.

5. The ratio of the lengths of the vectors, if such a constraint exists.

6. The number of sides on the fundamental domain, or tile (4 if it is a parallelogram, or 3 if it is a
   triangle created by slicing a parallelogram between the tips of its
   vectors).

7. The minimum number of fundamental domains necessary to create an arrangement which can be
   tiled using translation alone (primitive cell).

======= =============== ======== ================== =========== ===== =====
Number  Hermann-Mauguin Orbifold :math:`\cos\theta` :math:`a/b` Sides F.D.
======= =============== ======== ================== =========== ===== =====
1       p1              o                                       4     1
2       p2              2222                                    4     2
3       pm              \*\*     0                              4     2
4       pg              xx       0                              4     2
5       pmm             \*2222   0                              4     2
6       pmg             22\*     0                              4     4
7       pgg             22x      0                              4     4
8       cm              \*x      0                              4     2
9       cmm             2\*22    0                              3     4
10      p4              442      0                  1           4     4
11      p4m             \*442    0                  1           3     8
12      p4g             4\*2     0                  1           3     8
13      p3              333      1/2                1           4     3
14      p3m1            \*333    1/2                1           3     6
15      p31m            3\*3     -1/2               1           3     6
16      p6              632      -1/2               1           3     6
17      p6m             \*632    1/2                2           3     12
======= =============== ======== ================== =========== ===== =====

Renderings
^^^^^^^^^^

Individual renderings are available in ``/docs/wallpaper/renderings/``.
However, all of the groups are displayed below:

.. image:: _static/wallpaper.*
    :height: 300px

Representation within paccs
----------------------------------

Within the paccs code, each group is represented by a
:py:func:`paccs.wallpaper.WallpaperGroup` and can be created by
specification of the group's number, Hermann-Mauguin identifier, or orbifold
notation identifier.  The underlying group object returned stores information
that can be found in the above table as well as additional information related
to the symmetries present in the group.

A ``WallpaperGroup`` call actually returns a ``_WallpaperGroup`` tuple object.
This stores the ``number``, ``name``, ``symbol``, ``dot`` and ``ratio``
attributes with direct correspondence in the table above.  The ``half`` flag
indicates if the fundamental domain (tile) is a triangle instead of a parallelogram.  
The additional members are used to hold symmetry and stoichiometry data.

Under normal circumstances, it will never be necessary to interact with any of
these items: the number of wallpaper groups is finite and the list should not
require editing unless an error is present.  However, the format used for these
specifications is documented here for completeness.

Symmetry: ``corners`` and ``edges``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``corners`` attribute contains a list with zero or more elements.

* Each element will be a set containing two or more indices.  These indicate
  corners which must be identical: a particle placed on one of these corners
  must also appear on all other corners.

* Consider a fundamental domain (tile) with two vectors :math:`\vec a` and 
  :math:`\vec b`.  Corner 0 appears at :math:`\vec 0`, 1 at :math:`\vec a`, 2 
  at :math:`\vec b`, and 3 (if the tile is not a triangle) at :math:`\vec a+\vec b`.

The ``edges`` attribute contains a list with zero or more elements.

* Each element will be an ``_EdgeSymmetry`` tuple object with some number of
  edge indices.  These indicate edges having some relationship to each other
  with respect to symmetry.

  * If the element is an ``_FWD(i, j)``, this indicates that edges :math:`i`
    and :math:`j` must be identical in their forward directions.  Passing from
    the beginning to the end of :math:`i` must appear identical to passing from
    the beginning to the end of :math:`j`.  The directionality of the edges is
    defined below.
  
  * If the element is an ``_REV(i, j)``, this indicates that edges :math:`i`
    and :math:`j` must be identical in reverse.  Passing from the beginning to
    the end of :math:`i` must appear identical to passing from the end to the
    beginning of :math:`j`, and necessarily vice versa.  The directionality of
    the edges is defined below.
  
  * If the element is an ``_INV(i)``, this indicates that edge :math:`i` must
    have reflection symmetry.  Passing from the center to the beginning of
    :math:`i` must appear identical to passing from the center to the end of
    :math:`i`.  As the exact center is thus only required to be identical to
    itself, no constraints are placed on it.  This type of symmetry results
    because a 2-fold center of rotation exists at the center of the edge.

* Consider a fundamental domain (tile) with corners indexed as above.  Edge 0 
  proceeds from corner 0 to corner 1, 1 from 2 to 3, 2 from 0 to 2, 3 from 1 to 
  3, and 4 from 1 to 2. Edges 1 and 3 are only present if the tile is a parallelogram, 
  and edge 4 is only present if it is a triangle.  The indications of "from" and 
  "to" are important as they define the directionality of the edges.  A convention for
  edges to proceed from a corner with a lower index to a corner with a higher
  index has been used.

.. image:: _static/tiles.*
   :height: 500px

Stoichiometry: ``stoichiometry``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``stoichiometry`` attribute contains a list of fraction objects that are
used to account for corners when performing stoichiometry calculations.  Unlike
edge stoichiometry data which is generated dynamically, corner stoichiometry
data is held within.

This list should contain exactly one item for each corner, indicating the solid
fraction of a finite-sized particle that would be present within the boundary
of a fundamental domain (tile) if it was placed on that corner.  In the case of 
groups with any variable angles (:math:`\mathrm{p1}` and :math:`\mathrm{p2}`), the fraction is chosen for the case of
:math:`\cos\theta=0` and :math:`a/b=1`.  Due to symmetry, this choice is somewhat
arbitrary.

Tiling: ``vectors`` and ``copies``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These attributes indicate exactly how the tiling should take place to form a
translationally periodic cell.

* The ``vectors`` item contains a list of two tuples, each containing two
  values.  If this item is of the form :math:`\left[\left(s,t\right),
  \left(u,v\right)\right]`, and the fundamental domain (tile) has vectors :math:`\vec a` and
  :math:`\vec b`, then the translationally periodic cell will have vectors
  :math:`\vec c=s\vec a+t\vec b` and :math:`\vec d=u\vec a+v\vec b`.

* The ``copies`` item indicates the actual operations that should be performed
  during the tiling.  The presence of one item in this list causes one new fundamental 
  domain (tile) to be created at the origin of the translationally periodic cell.  The
  instructions within the item (which is a list itself) will then be executed
  to move the tile in space.  If no instructions are present (that is, if the
  item is an empty list), the tile will be left as is.  Possible instructions
  of the ``_PeriodicOperation`` tuple type are:

  * ``_REF(i, j)``: inversion on the vectors of the translationally periodic
    cell (not the tile).  :math:`i` and :math:`j` are flags for the two
    vectors.  When a flag is 0, no inversion occurs; when it is 1, inversion
    occurs along that vector.
  
  * ``_ROT(z)``: rotation by :math:`z/12` turns counterclockwise about the
    origin of the translationally periodic cell.  This convention was chosen
    for convenience since the definitions of all wallpaper groups require
    rotations which are integer multiples of :math:`30^\circ`, that is,
    :math:`\pi/6`, or a twelfth-turn.

  * ``_TRN(x, y)``: translation by :math:`x\vec c+y\vec d` (as defined in the
    previous list item).
