//! Skeletal animation: clips, sampling, playback, blending and state control.
//!
//! `Animator` drives playback, cross-fade, events and root motion. `sampler` holds
//! the pose math it drives, `clip` the data, `compression` the keyframe reduction
//! and `state_machine` the FSM, which returns the state change instead of reaching
//! into an animator.

pub mod clip;
pub mod compression;
pub mod convert;
pub mod sampler;
pub mod state_machine;

pub use clip::{
    AnimationEvent, Bone, BonePose, Clip, Interpolation, Keyframe, LoopMode, Pose, QuaternionTrack,
    Skeleton, Skin, VectorTrack,
};
pub use compression::{compress_clip, measure_clip, CompressSettings, CompressionStats};
pub use convert::{animator_from_model, select_clip};
pub use sampler::{
    blend_poses, build_child_lists, global_transforms, rest_pose, sample_clip, sample_pose,
    sample_position, sample_rotation, sample_scale, skin_matrices,
};
pub use state_machine::{
    AnimationStateMachine, Condition, ConditionOp, Parameter, State, StateChange, Transition,
};

use glam::{Mat4, Vec3};

/// Animation runtime over a skeleton, skins and clips handed over directly, so it
/// carries no asset or ECS dependency and the caller applies the result.
#[derive(Debug)]
pub struct Animator {
    skeleton: Skeleton,
    skins: Vec<Skin>,
    clips: Vec<Clip>,
    /// Per bone, its children. Rebuilt from the skeleton, never from a clip.
    child_bones: Vec<Vec<i32>>,
    rest_pose: Pose,
    current_pose: Pose,
    prev_pose: Pose,
    blended_pose: Pose,
    current_clip: i32,
    current_time: f32,
    playing: bool,
    time_scale: f32,
    loop_mode: LoopMode,
    reversed: bool,
    /// Bounce direction for `LoopMode::PingPong`.
    pingpong_forward: bool,
    skin_index: i32,
    /// `>= 0` while a cross-fade is running.
    prev_clip: i32,
    prev_time: f32,
    blend_duration: f32,
    blend_time: f32,
    blend_weight: f32,
    root_motion_enabled: bool,
    root_motion_delta: Vec3,
    /// Events crossed by the last `update_animation`, drained by the caller.
    events: Vec<AnimationEvent>,
    global_transforms: Vec<Mat4>,
    final_bone_matrices: Vec<Mat4>,
}

impl Animator {
    pub fn new(skeleton: Skeleton) -> Self {
        let child_bones = sampler::build_child_lists(&skeleton);
        let pose = sampler::rest_pose(&skeleton);
        let bone_count = skeleton.bones.len();
        let joint_count = skeleton.joint_nodes.len();
        Self {
            skeleton,
            skins: Vec::new(),
            clips: Vec::new(),
            child_bones,
            rest_pose: pose.clone(),
            current_pose: pose.clone(),
            prev_pose: pose.clone(),
            blended_pose: pose,
            current_clip: -1,
            current_time: 0.0,
            playing: true,
            time_scale: 1.0,
            loop_mode: LoopMode::Loop,
            reversed: false,
            pingpong_forward: true,
            skin_index: 0,
            prev_clip: -1,
            prev_time: 0.0,
            blend_duration: 0.0,
            blend_time: 0.0,
            blend_weight: 1.0,
            root_motion_enabled: false,
            root_motion_delta: Vec3::ZERO,
            events: Vec::new(),
            global_transforms: vec![Mat4::IDENTITY; bone_count],
            final_bone_matrices: vec![Mat4::IDENTITY; joint_count],
        }
    }

    pub fn skeleton(&self) -> &Skeleton {
        &self.skeleton
    }

    pub fn clips(&self) -> &[Clip] {
        &self.clips
    }

    pub fn set_clips(&mut self, clips: Vec<Clip>) {
        self.clips = clips;
        self.current_clip = -1;
        self.prev_clip = -1;
        self.current_time = 0.0;
    }

    pub fn skins(&self) -> &[Skin] {
        &self.skins
    }

    pub fn set_skins(&mut self, skins: Vec<Skin>) {
        self.skins = skins;
        self.skin_index = 0;
        self.final_bone_matrices = vec![Mat4::IDENTITY; self.skeleton.joint_nodes.len()];
    }

    // playback controls

    /// Advances the play head and re-evaluates the pose for this frame.
    pub fn update_animation(&mut self, dt: f32) {
        self.root_motion_delta = Vec3::ZERO;

        if self.clips.is_empty() || self.skeleton.root_bone_id == -1 {
            return;
        }
        let index = self.current_clip;
        if index < 0 || index as usize >= self.clips.len() {
            return;
        }

        let duration = self.clips[index as usize].duration;
        if !self.playing || duration <= 0.0 {
            return;
        }

        // Advance time, respecting speed, direction and loop mode.
        let effective_dt = dt * self.time_scale;
        let mut direction = if self.reversed { -1.0 } else { 1.0 };
        if self.loop_mode == LoopMode::PingPong && !self.pingpong_forward {
            direction = -direction;
        }

        let prev_time = self.current_time;
        let mut new_time = prev_time + effective_dt * direction;
        let moved_forward = direction > 0.0;
        let mut wrapped = false;

        match self.loop_mode {
            LoopMode::None => {
                if new_time >= duration {
                    new_time = duration;
                    self.playing = false;
                } else if new_time <= 0.0 {
                    new_time = 0.0;
                    self.playing = false;
                }
            }
            LoopMode::Loop => {
                while new_time >= duration {
                    new_time -= duration;
                    wrapped = true;
                }
                while new_time < 0.0 {
                    new_time += duration;
                    wrapped = true;
                }
            }
            LoopMode::PingPong => {
                // Reflect at the ends and flip direction; no loop wrap.
                while new_time >= duration {
                    new_time = 2.0 * duration - new_time;
                    self.pingpong_forward = !self.pingpong_forward;
                }
                while new_time < 0.0 {
                    new_time = -new_time;
                    self.pingpong_forward = !self.pingpong_forward;
                }
            }
        }
        self.current_time = new_time;

        self.fire_events(index as usize, prev_time, new_time, moved_forward);
        if self.root_motion_enabled {
            self.root_motion_delta = self.extract_root_motion(
                index as usize,
                prev_time,
                new_time,
                wrapped,
                moved_forward,
            );
        }

        let mut current = self.rest_pose.clone();
        {
            let clip = &self.clips[index as usize];
            sampler::sample_clip(clip, self.current_time, &mut current);
        }

        if self.prev_clip != -1 && self.blend_duration > 0.0 {
            let prev_index = self.prev_clip as usize;
            // The fading-out clip keeps advancing; Loop semantics so it never stalls.
            self.prev_time += effective_dt * direction;
            let prev_duration = self.clips[prev_index].duration;
            if prev_duration > 0.0 {
                while self.prev_time >= prev_duration {
                    self.prev_time -= prev_duration;
                }
                while self.prev_time < 0.0 {
                    self.prev_time += prev_duration;
                }
            }

            let mut previous = self.rest_pose.clone();
            {
                let clip = &self.clips[prev_index];
                sampler::sample_clip(clip, self.prev_time, &mut previous);
            }
            self.prev_pose = previous;

            self.blend_time += effective_dt.abs();
            let w = (self.blend_time / self.blend_duration).min(1.0);
            self.blend_weight = w * w * (3.0 - 2.0 * w); // smoothstep for a softer fade
            sampler::blend_poses(
                &self.prev_pose,
                &current,
                self.blend_weight,
                &mut self.blended_pose,
            );

            if self.blend_weight >= 1.0 {
                self.prev_clip = -1;
                self.blend_duration = 0.0;
            }
        } else {
            self.blended_pose.clone_from(&current);
        }
        self.current_pose = current;

        self.traverse_skeleton_roots();
    }

    pub fn play_animation(&mut self, index: i32) {
        if index < 0 || index as usize >= self.clips.len() {
            return;
        }
        self.current_clip = index;
        self.current_time = 0.0;
        self.prev_clip = -1; // a hard cut cancels any running fade
        self.blend_weight = 1.0;
        self.playing = true;
    }

    pub fn play_animation_by_name(&mut self, name: &str) -> bool {
        match self.clips.iter().position(|c| c.name == name) {
            Some(index) => {
                self.play_animation(index as i32);
                true
            }
            None => false,
        }
    }

    /// Resumes playback, restarting a clip that ran to the end under `LoopMode::None`.
    pub fn play(&mut self) {
        if self.current_clip == -1 {
            return;
        }
        let duration = self.duration();
        if !self.playing
            && duration > 0.0
            && self.current_time >= duration - f32::EPSILON
            && !self.reversed
        {
            self.current_time = 0.0;
        }
        self.playing = true;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.current_time = 0.0;
        self.prev_clip = -1;
        self.blend_weight = 1.0;
        self.blend_duration = 0.0;
        self.blend_time = 0.0;
        self.root_motion_delta = Vec3::ZERO;
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn set_time_scale(&mut self, scale: f32) {
        self.time_scale = scale;
    }

    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }

    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        self.loop_mode = mode;
    }

    pub fn loop_mode(&self) -> LoopMode {
        self.loop_mode
    }

    pub fn set_reversed(&mut self, reversed: bool) {
        self.reversed = reversed;
    }

    pub fn is_reversed(&self) -> bool {
        self.reversed
    }

    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    pub fn duration(&self) -> f32 {
        if self.current_clip < 0 {
            return 0.0;
        }
        self.clips
            .get(self.current_clip as usize)
            .map_or(0.0, |c| c.duration)
    }

    /// Normalised `[0, 1]` playback progress.
    pub fn normalized_time(&self) -> f32 {
        let duration = self.duration();
        if duration <= 0.0 {
            return 0.0;
        }
        self.current_time / duration
    }

    /// Jumps the play head and immediately evaluates the pose, for scrub controls
    /// and paused previewing.
    pub fn seek(&mut self, time: f32) {
        if self.skeleton.root_bone_id == -1 {
            return;
        }
        let index = self.current_clip;
        if index < 0 || index as usize >= self.clips.len() {
            return;
        }

        let duration = self.clips[index as usize].duration;
        self.current_time = if duration > 0.0 {
            time.clamp(0.0, duration)
        } else {
            0.0
        };

        let mut pose = self.rest_pose.clone();
        sampler::sample_clip(&self.clips[index as usize], self.current_time, &mut pose);
        self.current_pose = pose.clone();
        self.blended_pose = pose; // scrubbing shows the target clip only

        self.traverse_skeleton_roots();
    }

    // cross-fade blending

    /// Stops the current clip and cross-fades into the named one over `duration`
    /// seconds. Returns false when no clip carries that name.
    pub fn cross_fade_by_name(&mut self, name: &str, duration: f32) -> bool {
        match self.clips.iter().position(|c| c.name == name) {
            Some(index) => {
                self.cross_fade(index as i32, duration);
                true
            }
            None => false,
        }
    }

    pub fn cross_fade(&mut self, index: i32, duration: f32) {
        if index < 0 || index as usize >= self.clips.len() {
            return;
        }
        if self.current_clip == -1 {
            self.play_animation(index); // nothing to fade from
            return;
        }

        self.prev_clip = self.current_clip;
        self.prev_time = self.current_time;
        self.current_clip = index;
        self.current_time = 0.0;
        self.blend_duration = duration.max(f32::EPSILON);
        self.blend_time = 0.0;
        self.blend_weight = 0.0;
        self.playing = true;
    }

    pub fn is_blending(&self) -> bool {
        self.prev_clip != -1
    }

    /// 0 = source clip, 1 = target clip.
    pub fn blend_weight(&self) -> f32 {
        self.blend_weight
    }

    // events

    /// Events the last `update_animation` crossed. Handed over, not called back, so
    /// the runtime never re-enters gameplay code.
    pub fn take_events(&mut self) -> Vec<AnimationEvent> {
        std::mem::take(&mut self.events)
    }

    // root motion

    pub fn set_root_motion_enabled(&mut self, enabled: bool) {
        self.root_motion_enabled = enabled;
    }

    pub fn is_root_motion_enabled(&self) -> bool {
        self.root_motion_enabled
    }

    pub fn root_motion_delta(&self) -> Vec3 {
        self.root_motion_delta
    }

    // skin selection

    pub fn set_skin(&mut self, index: i32) {
        if index < 0 || index as usize >= self.skins.len() {
            return;
        }
        self.skin_index = index;
        let count = self.skins[index as usize].joint_nodes.len();
        self.final_bone_matrices = vec![Mat4::IDENTITY; count];
    }

    pub fn skin_index(&self) -> i32 {
        self.skin_index
    }

    /// World transforms times the selected skin's inverse binds, ready for the
    /// skinning path.
    pub fn final_bone_matrices(&self) -> &[Mat4] {
        &self.final_bone_matrices
    }

    /// The sampled and blended pose the matrices were built from.
    pub fn blended_pose(&self) -> &Pose {
        &self.blended_pose
    }

    pub fn has_animation(&self) -> bool {
        self.current_clip != -1
    }

    pub fn current_animation_index(&self) -> i32 {
        self.current_clip
    }

    // internals

    fn traverse_skeleton_roots(&mut self) {
        let roots: Vec<i32> = self
            .skeleton
            .bones
            .iter()
            .enumerate()
            .filter(|(_, bone)| bone.parent_id < 0)
            .map(|(i, _)| i as i32)
            .collect();

        for root in roots {
            sampler::calculate_bone_transform(
                &self.skeleton,
                &self.child_bones,
                &self.blended_pose,
                self.root_motion_enabled,
                root,
                Mat4::IDENTITY,
                &mut self.global_transforms,
            );
        }
        self.compute_final_matrices();
    }

    fn compute_final_matrices(&mut self) {
        let skin = if self.skin_index >= 0 {
            self.skins.get(self.skin_index as usize)
        } else {
            None
        };
        sampler::skin_matrices(
            &self.skeleton,
            &self.global_transforms,
            skin,
            &mut self.final_bone_matrices,
        );
    }

    /// Fires events whose time was crossed between `prev_time` and `new_time`,
    /// honouring play direction and loop wraps.
    fn fire_events(&mut self, index: usize, prev_time: f32, new_time: f32, moved_forward: bool) {
        let clip = &self.clips[index];
        if clip.events.is_empty() {
            return;
        }
        let duration = clip.duration;

        match self.loop_mode {
            LoopMode::Loop if moved_forward => {
                if new_time < prev_time {
                    // Wrapped forwards.
                    for event in &clip.events {
                        if event.time > prev_time && event.time <= duration {
                            self.events.push(event.clone());
                        }
                    }
                    for event in &clip.events {
                        if event.time >= 0.0 && event.time <= new_time {
                            self.events.push(event.clone());
                        }
                    }
                } else {
                    for event in &clip.events {
                        if event.time > prev_time && event.time <= new_time {
                            self.events.push(event.clone());
                        }
                    }
                }
            }
            LoopMode::Loop => {
                if new_time > prev_time {
                    // Wrapped backwards.
                    for event in &clip.events {
                        if event.time < prev_time && event.time >= 0.0 {
                            self.events.push(event.clone());
                        }
                    }
                    for event in &clip.events {
                        if event.time <= duration && event.time >= new_time {
                            self.events.push(event.clone());
                        }
                    }
                } else {
                    for event in &clip.events {
                        if event.time <= prev_time && event.time > new_time {
                            self.events.push(event.clone());
                        }
                    }
                }
            }
            // None / PingPong: fire across the traversed window regardless of
            // direction; reflections are approximated by the final segment.
            LoopMode::None | LoopMode::PingPong => {
                let a = prev_time.min(new_time);
                let b = prev_time.max(new_time);
                for event in &clip.events {
                    if event.time >= a && event.time <= b {
                        self.events.push(event.clone());
                    }
                }
            }
        }
    }

    /// The root joint's translation change between two sample times.
    fn extract_root_motion(
        &self,
        index: usize,
        prev_time: f32,
        new_time: f32,
        wrapped: bool,
        moved_forward: bool,
    ) -> Vec3 {
        let clip = &self.clips[index];
        let root_bone = self.skeleton.root_bone_id;
        let Some(track) = clip
            .position_tracks
            .iter()
            .find(|t| t.target_bone == root_bone)
        else {
            return Vec3::ZERO;
        };

        let prev = sampler::sample_position(prev_time, track);
        let next = sampler::sample_position(new_time, track);
        if !wrapped {
            return next - prev;
        }

        // Wrapped: accumulate the motion through the boundary ends.
        let end = sampler::sample_position(clip.duration, track);
        let start = sampler::sample_position(0.0, track);
        if moved_forward {
            (end - prev) + (next - start)
        } else {
            (start - prev) + (next - end)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::clip::{Bone, Keyframe};

    fn two_bone_skeleton() -> Skeleton {
        let mut skeleton = Skeleton::new();
        let root = Bone {
            id: 0,
            local_transform: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            ..Default::default()
        };
        let child = Bone {
            id: 1,
            parent_id: 0,
            local_transform: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            ..Default::default()
        };
        skeleton.bones.push(root);
        skeleton.bones.push(child);
        skeleton.joint_nodes.push(1);
        skeleton.root_bone_id = 0;
        skeleton
    }

    fn sliding_clip(name: &str, duration: f32) -> Clip {
        let mut clip = Clip::new(name);
        clip.position_track_mut(0)
            .set_keys(&[0.0, duration], &[Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0)]);
        clip.update_duration();
        clip
    }

    #[test]
    fn a_looping_clip_wraps_past_its_end() {
        let mut animator = Animator::new(two_bone_skeleton());
        animator.set_clips(vec![sliding_clip("run", 1.0)]);
        animator.play_animation(0);
        animator.update_animation(1.5);
        assert!((animator.current_time() - 0.5).abs() < 1e-6);
        assert!(animator.is_playing());
    }

    #[test]
    fn a_non_looping_clip_stops_at_its_end() {
        let mut animator = Animator::new(two_bone_skeleton());
        animator.set_clips(vec![sliding_clip("run", 1.0)]);
        animator.set_loop_mode(LoopMode::None);
        animator.play_animation(0);
        animator.update_animation(3.0);
        assert_eq!(animator.current_time(), 1.0);
        assert!(!animator.is_playing());
        assert_eq!(animator.normalized_time(), 1.0);
    }

    #[test]
    fn the_played_clip_reaches_its_last_key() {
        let mut animator = Animator::new(two_bone_skeleton());
        animator.set_clips(vec![sliding_clip("run", 1.0)]);
        animator.play_animation(0);
        animator.seek(1.0);
        assert_eq!(
            animator.blended_pose().bones[0].translation,
            Vec3::new(0.0, 2.0, 0.0)
        );
        assert_eq!(
            animator.final_bone_matrices()[0].w_axis.truncate(),
            Vec3::new(0.0, 3.0, 0.0)
        );
    }

    #[test]
    fn a_cross_fade_ramps_the_weight_to_one() {
        let mut animator = Animator::new(two_bone_skeleton());
        animator.set_clips(vec![sliding_clip("a", 1.0), sliding_clip("b", 1.0)]);
        animator.play_animation(0);
        animator.cross_fade(1, 1.0);
        assert!(animator.is_blending());
        assert_eq!(animator.blend_weight(), 0.0);
        animator.update_animation(1.0);
        assert_eq!(animator.blend_weight(), 1.0);
        assert!(!animator.is_blending());
    }

    #[test]
    fn events_crossed_by_the_play_head_are_handed_over() {
        let mut clip = sliding_clip("run", 1.0);
        clip.events.push(AnimationEvent::new(0.5, "footstep"));
        let mut animator = Animator::new(two_bone_skeleton());
        animator.set_clips(vec![clip]);
        animator.play_animation(0);

        animator.update_animation(0.25);
        assert!(animator.take_events().is_empty());
        animator.update_animation(0.5);
        let events = animator.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "footstep");
    }

    #[test]
    fn a_single_key_track_is_constant() {
        let mut clip = Clip::new("pose");
        clip.position_track_mut(0)
            .keyframes
            .push(Keyframe::new(0.0, Vec3::X));
        clip.duration = 1.0;
        let mut animator = Animator::new(two_bone_skeleton());
        animator.set_clips(vec![clip]);
        animator.play_animation(0);
        animator.update_animation(0.0);
        assert_eq!(animator.blended_pose().bones[0].translation, Vec3::X);
    }
}
