"""Independent mesh/box intersection checks on the simulator collision meshes.

The triangle test uses all 13 separating axes. Containment is checked with a
convex upper bound, so concave containment can conservatively reject a case.
This checks sampled poses; it does not by itself certify continuous clearance.
"""
import numpy as np
from scipy.spatial import ConvexHull

import bruce_env as be


def mesh_box_overlap(vertices, faces, center, rotation, half, margin=0.):
    points=(np.asarray(vertices)-center)@rotation
    half=np.asarray(half)+margin
    if np.any(points.min(0)>half+1e-10) or np.any(points.max(0)<-half-1e-10):return False
    if np.any(np.all(np.abs(points)<=half+1e-10,axis=1)):return True
    triangles=points[np.asarray(faces)]
    keep=np.all(triangles.min(1)<=half+1e-10,axis=1)&np.all(triangles.max(1)>=-half-1e-10,axis=1)
    triangles=triangles[keep]
    if len(triangles):
        edges=np.roll(triangles,-1,axis=1)-triangles
        axes=np.concatenate((np.broadcast_to(np.eye(3),(len(triangles),3,3)),
            np.cross(edges[:,0],edges[:,1])[:,None,:],
            np.cross(edges[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(-1,9,3)),axis=1)
        projection=np.einsum('ntc,nac->nat',triangles,axes)
        radius=np.abs(axes)@half
        separated=(projection.min(2)>radius+1e-10)|(projection.max(2)<-radius-1e-10)
        if np.any(~np.any(separated,axis=1)):return True
    if len(points)<4 or np.linalg.matrix_rank(points-points[0])<3:return False
    hull=ConvexHull(points)
    return bool(np.all(hull.equations[:,3]<=1e-10))


class CollisionAudit:
    def __init__(self,scene):
        self.meshes=[]
        for i in range(scene.robot.geom_start,scene.robot.geom_end):
            geom=scene.scene.rigid_solver.geoms[i]
            # init vertices are in the geometry frame, init pose is in its link.
            vertices=np.asarray(geom.init_verts)@be.quat_to_mat(geom.init_quat).T+geom.init_pos
            self.meshes.append(dict(name=geom.link.name,index=scene.link_index[geom.link.name],
                vertices=vertices,faces=np.asarray(geom.init_faces),geom_index=i))

    def check_transform(self,scene):
        pos=be.to_np(scene.robot.get_links_pos(relative=False))
        quat=be.to_np(scene.robot.get_links_quat(relative=False))
        peak=0.
        for mesh in self.meshes:
            i=mesh['index'];world=mesh['vertices']@be.quat_to_mat(quat[i]).T+pos[i]
            actual=be.to_np(scene.scene.rigid_solver.geoms[mesh['geom_index']].get_verts())
            peak=max(peak,float(np.max(np.abs(world-actual))))
        if peak>1e-5:raise RuntimeError(f'collision mesh transform mismatch: {peak}')
        return peak

    def intersections(self,scene,box,half,margin=0.):
        pos=be.to_np(scene.robot.get_links_pos(relative=False))
        quat=be.to_np(scene.robot.get_links_quat(relative=False))
        center=scene.box_pos(box);rotation=be.quat_to_mat(be.to_np(box.get_quat()))
        result=[]
        for mesh in self.meshes:
            i=mesh['index'];world=mesh['vertices']@be.quat_to_mat(quat[i]).T+pos[i]
            if mesh_box_overlap(world,mesh['faces'],center,rotation,half,margin):result.append(mesh['name'])
        return result


def robot_box_contacts(scene,box):
    raw=scene.robot.get_contacts(exclude_self_contact=True)
    contacts=None if raw is None else {k:be.to_np(v) for k,v in raw.items()}
    if contacts is None:return 0
    robot=set(range(scene.robot.geom_start,scene.robot.geom_end))
    obstacle=set(range(box.geom_start,box.geom_end))
    return sum((int(a) in robot and int(b) in obstacle) or (int(b) in robot and int(a) in obstacle)
               for a,b in zip(contacts['geom_a'],contacts['geom_b']))
