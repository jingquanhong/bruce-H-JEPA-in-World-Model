import numpy as np
import trimesh
from collision_audit import mesh_box_overlap


def test_mesh_box_separation_touch_and_containment():
    mesh=trimesh.creation.box(extents=(2,2,2))
    def hit(center,half=(.1,.1,.1)):
        return mesh_box_overlap(mesh.vertices,mesh.faces,np.array(center),np.eye(3),half)
    assert hit([0,0,0])  # box enclosed by robot mesh
    assert hit([0,0,0],(2,2,2))  # mesh enclosed by box
    assert hit([1.1,0,0])  # touching counts as intersection
    assert not hit([1.10001,0,0])


def test_triangle_separating_axis_rejects_overlapping_bounds():
    # Both AABBs overlap, but the triangle stays outside the cube corner.
    vertices=np.array([[.2,2,0],[2,.2,0],[2,2,0]])
    assert not mesh_box_overlap(vertices,[[0,1,2]],[0,0,0],np.eye(3),[1,1,1])
    assert mesh_box_overlap(vertices,[[0,1,2]],[0,0,0],np.eye(3),[1.2,1.2,1])


def test_rotated_box_and_margin():
    mesh=trimesh.creation.box(extents=(.2,.2,.2))
    angle=np.pi/4
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    assert not mesh_box_overlap(mesh.vertices,mesh.faces,[.3,0,0],rotation,[.1,.1,.1])
    assert mesh_box_overlap(mesh.vertices,mesh.faces,[.3,0,0],rotation,[.1,.1,.1],margin=.05)
