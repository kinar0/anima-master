"""Shared writer guidance; visibility stays a per-instance semantic decision."""

APPEARANCE_VISIBILITY_POLICY = """

Visible appearance: follow the user's specified composition, or your chosen
composition where unspecified. For each instance, first determine which body parts
appear within the frame. Then describe only their actually visible appearance,
considering viewpoint, pose, clothing, occlusion and reflections. A body part being
in frame does not make every feature of that part visible. Omit features that are
outside the frame, hidden or not discernible; retain the visible shapes and details.
Apply this throughout tags, prose and priority tags. Saved profiles supply facts,
not a checklist to reproduce. Do not change the composition or pose to expose a
saved feature, or reintroduce an invisible feature in another field. Evaluate each
person and each panel independently, including repeated identities. Keep canonical
identity, and obey user edits and profile facts for the features you describe.
"""
