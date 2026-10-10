"""Basket contact-v2 RGB policy; only new candidate actions use physical gripper mapping."""
# Inference functions from recovered_basket base_work and contact_v2 sources.
import numpy as np

CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
CLASSES = ('background','milk','basket','left_gripper','right_gripper','robot','milk_contact','basket_handle')
PALETTE = {1:(255,64,64),2:(64,128,255),3:(64,220,128),4:(255,192,64),6:(255,64,192),7:(64,255,255)}
TASK = 'Pick the bottle then place it to the basket, carry the basket with the other arm.'
GRIPPERS = [6,13]
INTERPOLATION = 'INTER_AREA'


def enhance_rgb(rgb,mask):
    rgb=np.asarray(rgb);mask=np.asarray(mask)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or mask.shape!=rgb.shape[:2]:
        raise ValueError('Expected aligned RGB uint8 and label map')
    if mask.dtype.kind not in 'iu' or not mask.size or mask.min()<0 or mask.max()>=len(CLASSES):
        raise ValueError('Unknown contact semantic label')
    output=rgb.astype(np.float32)
    gray=output@np.array([.299,.587,.114],np.float32)
    background=mask==0;output[background]=.5*output[background]+.5*gray[background,None]
    for label,color in PALETTE.items():
        selected=mask==label;alpha=.45 if label in (6,7) else .25
        output[selected]=(1-alpha)*output[selected]+alpha*np.array(color,np.float32)
    return np.clip(np.rint(output),0,255).astype(np.uint8)


enhance = enhance_rgb


def policy_arrays(images, state):
    state = np.asarray(state,dtype=np.float32)
    if len(images)!=3 or state.shape!=(14,) or not np.isfinite(state).all():
        raise ValueError('Expected three RGB cameras and finite14 state')
    result = {'observation.state':state, 'task':TASK}
    for camera,rgb in zip(CAMERAS,images):
        if rgb.dtype!=np.uint8 or rgb.shape!=(288,384,3):
            raise ValueError('Expected RGB uint8 HWC384x288')
        result['observation.images.'+camera]=rgb.transpose(2,0,1).astype(np.float32)/255
    return result


def _validated(actions, limits):
    result=np.asarray(actions,dtype=np.float32).copy()
    limits=np.asarray(limits,dtype=np.float32)
    if result.ndim not in (1,2) or result.shape[-1]!=14 or not np.isfinite(result).all():
        raise ValueError('Expected finite14 or Nx14 actions')
    if limits.shape!=(2,2) or not np.isfinite(limits).all() or np.any(limits[:,1]<=limits[:,0]):
        raise ValueError('Invalid gripper limits')
    return result,limits


def physical_actions(actions, limits):
    result,limits=_validated(actions,limits)
    result[...,GRIPPERS]=limits[:,0]+np.clip(result[...,GRIPPERS],0,1)*(limits[:,1]-limits[:,0])
    return result


denormalize = physical_actions


class FullChunk:
    def __init__(self,predict):self.predict=predict;self.reset()

    def reset(self):self.actions=None;self.index=50

    def next(self,observation):
        if self.index==50:
            actions=np.asarray(self.predict(observation),dtype=np.float32)
            if actions.shape!=(50,14) or not np.isfinite(actions).all():
                raise ValueError('Expected finite 50x14 contact action chunk')
            self.actions=actions.copy();self.index=0
        action=self.actions[self.index].copy();self.index+=1
        return action
