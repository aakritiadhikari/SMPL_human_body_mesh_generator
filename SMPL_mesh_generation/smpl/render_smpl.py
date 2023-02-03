import cv2
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from serialization import load_model
from scipy.io import loadmat

## Load SMPL model (here we load the neural model)
m = load_model('../models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl')
n = loadmat('../cross_leg/pressure14.mat')
beta = n['body_shape']
beta = np.resize(beta,(10,))
pose = n['joint_angles']
pose = np.resize(pose,(72,))

## Assign random pose and shape parameters
#m.pose[:] = np.random.rand(m.pose.size) * .2 #size(72,)
#m.betas[:] = np.random.rand(m.betas.size) * .03 #size(10,)
m.pose[:] = pose
m.betas[:] = beta
m.pose[0] = np.pi

#print(m.pose.shape)
#exit(0)
## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

## Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=np.ones_like(m)*.9,
    light_color=np.array([1., 1., 1.]))


cv2.imshow('render_SMPL', rn.r)
cv2.waitKey(0)
cv2.destroyAllWindows()
