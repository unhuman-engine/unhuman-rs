//! Collision filtering: the scene's `CollisionFilter` as Rapier interaction
//! groups.

use rapier3d::prelude::{Group, InteractionGroups, InteractionTestMode};

use crate::components::CollisionGroups;

/// Rapier's two-mask test. Unnamed bits pass through: the component is a raw
/// `u32` bitmask rather than a fixed enum, so a scene can use all 32 groups.
pub(crate) fn interaction_groups(groups: &CollisionGroups) -> InteractionGroups {
    InteractionGroups::new(
        Group::from_bits_retain(groups.memberships),
        Group::from_bits_retain(groups.filter),
        InteractionTestMode::And,
    )
}
