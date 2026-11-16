"""Independent planning model built from the installed BRUCE model export.

Pinocchio is used only on the external development computer. No simulation
files are changed. Model equivalence to the Khadas binary must pass first.
"""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin


def build(path):
    description = json.loads(Path(path).read_text())
    model = pin.Model()
    root = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "base")
    model.appendBodyToJoint(root, pin.Inertia(description["mass"][0],
        np.array(description["com"][0]), np.array(description["inertia_com"][0])), pin.SE3.Identity())
    for side, start in [("right",1),("left",6)]:
        parent = root
        for j in range(5):
            i = start+j
            rotation = np.eye(3)
            if j==1:rotation=pin.rpy.rpyToMatrix(-np.pi/2,0,0)
            if j==2:rotation=pin.rpy.rpyToMatrix(np.pi/2,0,0)
            placement = pin.SE3(rotation, np.array(description["coordinate_vectors"][i]))
            parent=model.addJoint(parent,pin.JointModelRZ(),placement,f"{side}_{j}")
            model.appendBodyToJoint(parent,pin.Inertia(description["mass"][i],
                np.array(description["com"][i]),np.array(description["inertia_com"][i])),pin.SE3.Identity())
        for name, y in [("toe",description["at"]),("heel",-description["ah"])]:
            model.addFrame(pin.Frame(f"{side}_{name}",parent,0,
                pin.SE3(np.eye(3),np.array([description["a5"],y,0.])),pin.FrameType.OP_FRAME))
    return model,description


def check(path, out=None):
    model,d=build(path)
    errors={"H":0.,"CG":0.,"AG":0.,"com":0.,"foot":0.}
    # Source convention angular/linear, Pinocchio convention linear/angular.
    order=[3,4,5,0,1,2]+list(range(6,16))
    momentum_order=[3,4,5,0,1,2]
    for s in d["samples"]:
        data=model.createData();q=pin.neutral(model);q[:3]=[0,0,.43];q[7:]=s["q"]
        v=np.r_[np.zeros(6),s["dq"]]
        pin.computeAllTerms(model,data,q,v)
        H=(np.triu(data.M)+np.triu(data.M,1).T)[np.ix_(order,order)]
        CG=data.nle[order]
        AG=pin.computeCentroidalMap(model,data,q)[np.ix_(momentum_order,order)]
        com=pin.centerOfMass(model,data,q)
        for name,actual in [("H",H),("CG",CG),("AG",AG),("com",com)]:
            errors[name]=max(errors[name],float(np.max(abs(actual-np.array(s[name])))))
        pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
        for name in ["right_toe","right_heel","left_toe","left_heel"]:
            foot=data.oMf[model.getFrameId(name)].translation-np.array([0,0,.43])
            errors["foot"]=max(errors["foot"],float(np.max(abs(foot-np.array(s[name+"_body"])))))
    passed=all(x<1e-6 for x in errors.values())
    report=dict(status="PASS" if passed else "FAIL",samples=len(d["samples"]),max_abs_errors=errors,
                hardware_actuation=False,existing_simulation_modified=False)
    output=Path(out) if out is not None else Path(path).with_name("planning-model-equivalence.json")
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if not passed:raise SystemExit(1)


if __name__=="__main__":
    check(Path(__file__).with_name("model-export.json"))
