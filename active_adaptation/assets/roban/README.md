# Roban S22 handball fixed-head asset

This directory contains the minimum mesh and MJCF subset needed by the training
adapter. It was copied from the local `roban_s22_0901.zip` delivery. The source
package identifies the model as `roban_s22_handball`, version `1.0.0`, derived
from the Kuavo `biped_s17` model. Its `package.xml` declares the BSD license.

Authoritative training model:

- `mjcf/biped_s17_verified_handball_fixedhead.xml`
- 28 STL meshes referenced by that MJCF under `meshes/`
- `package.xml`, retained only for the delivered version/license metadata

The source MJCF is kept physically unchanged. In particular, mass, inertia,
joint axes/ranges, collision shapes and source actuator order are preserved.
Its geoms have no XML names, so `active_adaptation.assets.roban._get_spec()`
assigns deterministic in-memory names before applying collision configuration.
It also removes the source torque motors in memory because the training entity
creates its own `BuiltinPositionActuatorCfg` actuators.

MuJoCo compilation on 2026-09-03 produced a total body mass of
`40.69991336 kg`. This supersedes the older mass stated in the delivered README.
The URDF and scene file are intentionally omitted because training does not
depend on them.
