"""Near-sole (0.1 mm band) polygons projected to a flat planning plane.

The curved sole and a tilted foot need actual contact checks separately.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull
import trimesh


class SoleGeometry:
    def __init__(self, urdf, level_foot_poses):
        path=Path(urdf); root=ET.parse(path).getroot()
        self.vertices=[]
        self.level_foot_poses=level_foot_poses
        for side,pose in zip(('r','l'),level_foot_poses):
            name='ankle_pitch_link_'+side
            link=next(l for l in root.findall('link') if l.get('name')==name)
            collision=link.find('collision'); mesh=collision.find('geometry/mesh')
            vertices=np.asarray(trimesh.load(path.parent/mesh.get('filename')).vertices)
            vertices=vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            origin=collision.find('origin')
            if origin is not None:
                xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
                vertices=vertices@pin.rpy.rpyToMatrix(rpy).T+xyz
            z=(vertices@pose.rotation.T)[:,2]
            self.vertices.append(vertices[z<z.min()+.0001])

    def polygon(self, side, pose):
        points=self.vertices[side]@pose.rotation.T+pose.translation
        hull=ConvexHull(points[:,:2])
        return points[hull.vertices,:2]

    def wrench_rows(self, side):
        """A*w <= 0 for a flat-sole COP; w is world wrench at ankle."""
        vertices=self.vertices[side]@self.level_foot_poses[side].rotation.T
        h=-float(np.mean(vertices[:,2]))
        rows=[]
        for ax,ay,b in ConvexHull(vertices[:,:2]).equations:
            rows.append([-ax*h,-ay*h,b,ay,-ax,0.])
        return np.asarray(rows)


def margin(point, polygon):
    hull=ConvexHull(np.asarray(polygon))
    return float(np.min(-(hull.equations[:,:2]@point+hull.equations[:,2])))
